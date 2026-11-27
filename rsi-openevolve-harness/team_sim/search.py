"""Optional OpenEvolve search for a round that asked for one (v3 red/yellow), with ACE and the preset skills.

Off by default: the simulator only records search_requested. Turn it on with --search evolve.
The search runs scripts/run_evolve.py (unchanged) on the round's files, starting from the cap-1.0
policy and from the deployed checkpoint; its best policy then goes through the same gate as every
other candidate. The model key is never passed on the command line: the default OpenEvolve config
(configs/openevolve.split-v1.yaml) reads it from the OPENEVOLVE_API_KEY environment variable.

ACE: --ace team_ace (default) gives the stage a team_ace scope below state/sim/<run>/<fold-protocol-policy>
(see Search.ace_root, so runs and policies never share ACE state) and seeds the preset
skills there; --ace-mode decides whether reflection may call a model (disabled / fixture / live, the
last one also needs --ace-approve-live and a call budget). --ace stub uses harness.ace_stub and a
plain playbook file. Records are enqueued for ACE after the search (no model call); with
--ace-worker-seconds > 0 (and mode fixture/live) the bounded team_ace worker then processes up to
--ace-max-jobs queued jobs (one Reflector + one Curator call each), each run limited to that many
seconds. team_ace scopes are per data/checkpoint, so every round gets a new scope; candidate
entries that ACE wrote in the previous search of the same run are carried into the next round's
scope (marked carried_from), otherwise nothing ACE learns would ever reach a later prompt.
Each ACE call waits at most --ace-request-timeout seconds and may write --ace-max-tokens tokens
(team_ace's own defaults, 20 s / 1024, are too small for a reasoning model such as Qwen3.8-27B).
--ace-no-thinking sends enable_thinking=false with every ACE call: on pytrio, Qwen3.8-27B otherwise
thinks first, and on the long ACE prompts the thinking used up the whole token cap, leaving an empty
answer (all 10 ACE calls of the 2026-11-24 captain run). `python -m team_sim ace-retry` reprocesses
the queued ACE jobs of a finished round on a copy of its state (team_sim/ace_retry.py).
The round's playbook shows the preset skills first and then at most --ace-max-shown (default 3)
learned entries, newest first, within 2400 characters (team_ace.store.render_settings); before
2026-11-24 the entries were shown in hash-ID order up to 1900 characters, so carried entries pushed
the preset skills out.

Time limit (--search-timeout): run_evolve.py runs in its own process group. At the limit its main
process gets SIGTERM, which OpenEvolve handles by stopping, saving the best program of the finished
iterations and exiting normally, so a slow search still yields a candidate. Whatever is left of the
group afterwards (OpenEvolve pool workers outlive a killed parent and keep calling the model) is
killed. A best program identical to the starting policy is not a new candidate and is not retrained.
"""
import contextlib
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from harness import PROJECT_ROOT

from . import seeds as preset


def _exited(pid, seconds):
    """Wait up to `seconds` for child `pid` to exit, without reaping it.

    An exited but unreaped child keeps its pid, so its process-group id cannot be handed to an
    unrelated process before the group's leftovers are killed.
    """
    deadline = time.monotonic() + seconds
    while os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT) is None:
        left = deadline - time.monotonic()
        if left <= 0:
            return False
        time.sleep(min(1.0, left))
    return True


def run_in_group(command, seconds, grace=90, **popen):
    """Run `command` in a new process group with a soft time limit; returns (returncode, stopped).

    stopped is None (finished in time), "time_limit" (SIGTERM at the limit, then exited on its own)
    or "killed" (still alive `grace` seconds after SIGTERM). Leftovers of the group are always killed.
    """
    process = subprocess.Popen(command, start_new_session=True, **popen)
    stopped = None
    try:
        if not _exited(process.pid, seconds):
            stopped = "time_limit"
            os.kill(process.pid, signal.SIGTERM)
            if not _exited(process.pid, grace):
                stopped = "killed"
                os.killpg(process.pid, signal.SIGKILL)
    finally:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(process.pid, signal.SIGKILL)
        code = process.wait()
    return code, stopped


class Search:
    def __init__(self, api_base, model, iterations=6, workers=1, oe_config=None, timeout=1800, grace=90,
                 ace="team_ace", ace_mode="disabled", ace_max_calls=0, ace_approve_live=False,
                 ace_model=None, ace_api_base=None, ace_api_key_env=None, ace_worker_seconds=0,
                 ace_max_jobs=3, ace_request_timeout=300, ace_max_tokens=4096, ace_thinking=None,
                 ace_max_shown=3, seeds_path=None, protocol=None):
        self.api_base, self.model, self.iterations, self.workers = api_base, model, int(iterations), int(workers)
        self.oe_config = str(oe_config or PROJECT_ROOT / "configs" / "openevolve.split-v1.yaml")
        self.timeout, self.grace, self.ace, self.ace_mode = int(timeout), int(grace), ace, ace_mode
        self.ace_max_calls, self.ace_approve_live = int(ace_max_calls), bool(ace_approve_live)
        self.ace_model, self.ace_api_base, self.ace_api_key_env = ace_model, ace_api_base, ace_api_key_env
        self.ace_worker_seconds, self.seeds_path, self.protocol = int(ace_worker_seconds), seeds_path, protocol
        self.ace_max_jobs = int(ace_max_jobs)
        self.ace_request_timeout, self.ace_max_tokens = int(ace_request_timeout), int(ace_max_tokens)
        self.ace_thinking = ace_thinking  # None: send nothing; True/False: enable_thinking with each ACE call
        self.ace_max_shown = int(ace_max_shown)
        self.last = {}               # outcome of the latest search, copied into the round's decision
        self._previous = {}          # run dir -> config of the previous search in that run (ACE carry-over)
        if ace_mode == "live" and not ace_approve_live:
            raise ValueError("ACE live mode needs --ace-approve-live (it spends model quota)")
        if not 5 <= self.ace_request_timeout <= 900:
            raise ValueError("--ace-request-timeout must be within 5..900 seconds")
        if not 256 <= self.ace_max_tokens <= 16384:
            raise ValueError("--ace-max-tokens must be within 256..16384")
        if ace_thinking not in (None, True, False):
            raise ValueError("ace_thinking must be None, True or False")
        if not 0 <= self.ace_max_shown <= 20:
            raise ValueError("--ace-max-shown must be within 0..20")
        need = 2 * self.ace_request_timeout + 30
        if ace_mode == "live" and 0 < self.ace_worker_seconds < need:
            raise ValueError(f"--ace-worker-seconds must be at least {need}: one ACE job is a Reflector and a "
                             f"Curator call of up to --ace-request-timeout={self.ace_request_timeout}s each")
        if "${OPENEVOLVE_API_KEY}" in Path(self.oe_config).read_text(encoding="utf-8") and not os.environ.get("OPENEVOLVE_API_KEY"):
            raise ValueError(f"{self.oe_config} reads the model key from $OPENEVOLVE_API_KEY; set it in the "
                             "environment first (any value works for scripts/mock_llm_server.py)")

    @staticmethod
    def ace_root(sim):
        """ACE storage of one simulated run: state/sim/<out name>-<path hash>/f<fold>-p<protocol>-<policy>.

        team_ace keys a scope by data and checkpoint only, so without this a later run (or another
        policy that deploys the same checkpoint) would reopen the same scope with its spent call
        budget, old queue and entries.
        """
        out = Path(sim.out).resolve()
        tag = f"{out.name}-{hashlib.sha256(str(out).encode('utf-8')).hexdigest()[:6]}"
        return PROJECT_ROOT / "state" / "sim" / tag / "-".join(Path(sim._run_dir).resolve().relative_to(out).parts)

    def _run_config(self, sim, k, incumbent, work):
        config = {"task_name": sim.task, "stage_id": f"sim-{sim.task}-{work.parent.parent.name}-r{k:02d}",
                  "train_path": sim.files["train"], "validation_path": sim.files["validation"],
                  "parent_checkpoint": str(incumbent), "work_dir": str(work / "work"), "seed": 42,
                  "limits": {"max_epochs": sim.max_epochs, "train_timeout_s": 300, "cpu_threads": 1},
                  "split": {"min_groups": 4}, "records": {"include_prompt": True},
                  "prompt": {"playbook_max_chars": 2500, "recent_failures": 3}}
        if self.ace == "team_ace":
            from team_ace.store import digest, scope_for
            root = self.ace_root(sim)
            config["ace"] = {"module": "team_ace", "batch_size": 1, "mode": self.ace_mode, "storage_root": str(root),
                             "record_origin": "simulation", "max_total_calls": self.ace_max_calls,
                             "live_calls_approved": self.ace_approve_live, "model": self.ace_model,
                             "api_base": self.ace_api_base, "api_key_env": self.ace_api_key_env,
                             "request_timeout_s": self.ace_request_timeout, "max_tokens": self.ace_max_tokens}
            if self.ace_thinking is not None:
                config["ace"]["enable_thinking"] = self.ace_thinking
            # Preset skills first, then the newest learned entries; 2400 stays below the 2500-character
            # playbook limit of the search prompt (prompt.playbook_max_chars).
            config["ace"]["render"] = {"base_first": True, "max_learned": self.ace_max_shown, "max_chars": 2400}
            config["ace"]["scope"] = scope_for(config)
            config["playbook_path"] = str(root / sim.task / digest(config["ace"]["scope"])[:16] / "playbook.md")
        else:
            config["ace"] = {"module": "harness.ace_stub", "batch_size": 10}
            config["playbook_path"] = str(work / "work" / "playbook" / "playbook.md")
        return config

    def run(self, sim, k, incumbent, initial_policy_source):
        work = sim._run_dir / f"round-{k:02d}" / "search"
        work.mkdir(parents=True, exist_ok=True)
        config = self._run_config(sim, k, incumbent, work)
        run_path = work / "run.json"
        run_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        context = {"task_name": sim.task, "task_type": "regression", "protocol": self.protocol or sim.protocol}
        seeded = preset.seed(config, context, self.seeds_path)
        if self.ace == "team_ace":
            seeded["carried"] = self._carry_over(self._previous.get(str(sim._run_dir)), config)
        (work / "seeded.json").write_text(json.dumps(seeded, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        policy = work / "initial_policy.py"
        policy.write_text(initial_policy_source, encoding="utf-8")
        command = [sys.executable, "-B", str(PROJECT_ROOT / "scripts" / "run_evolve.py"), "--run-config", str(run_path),
                   "--oe-config", self.oe_config, "--initial-policy", str(policy), "--iterations", str(self.iterations),
                   "--workers", str(self.workers), "--api-base", self.api_base, "--model", self.model]
        started = time.monotonic()
        with (work / "evolve.log").open("w", encoding="utf-8") as log:
            code, stopped = run_in_group(command, self.timeout, grace=self.grace, cwd=str(PROJECT_ROOT),
                                         stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
        best = Path(config["work_dir"]) / "openevolve_output" / "best" / "best_program.py"
        source = best.read_text(encoding="utf-8") if code == 0 and best.is_file() else None
        same = source is not None and source.strip() == initial_policy_source.strip()
        result = {"returncode": code, "stopped": stopped, "seconds": round(time.monotonic() - started, 1),
                  "time_limit": self.timeout, "best_program": source is not None, "best_is_initial": same}
        self.last = dict(result)
        if self.ace == "team_ace":
            try:
                from team_ace.store import flush
                result["ace_enqueued"] = flush(config)
                if self.ace_worker_seconds > 0 and self.ace_mode in ("fixture", "live"):
                    from team_ace.__main__ import bounded_worker
                    from team_ace.store import status
                    result["ace_worker"] = []
                    for _ in range(self.ace_max_jobs):
                        if not status(config)["jobs"].get("pending", 0):
                            break
                        result["ace_worker"].append(bounded_worker(str(run_path), seconds=self.ace_worker_seconds))
                    result["ace_status"] = status(config)
                self._previous[str(sim._run_dir)] = config
            except Exception as error:  # ACE problems must not stop the simulation
                result["ace_error"] = f"{type(error).__name__}: {error}"
        (work / "search_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str) + "\n",
                                                 encoding="utf-8")
        return None if same else source

    def _carry_over(self, previous, config):
        """Copy ACE-written candidate entries of the previous round's scope into this round's scope."""
        if not previous:
            return []
        from team_ace.store import connect, digest, encoded, exclusive, publish
        old_db = connect(previous)
        try:
            bodies = [json.loads(row["body"]) for row in old_db.execute("SELECT body FROM entries ORDER BY id")]
        finally:
            old_db.close()
        keep = [b for b in bodies if not b.get("base_experience") and b.get("status") == "candidate"
                and b.get("mode") == self.ace_mode]
        if not keep:
            return []
        scope_id, source = digest(config["ace"]["scope"]), digest(previous["ace"]["scope"])[:16]
        with exclusive(config):
            db = connect(config)
            try:
                with db:
                    for body in keep:
                        body = dict(body, scope_id=scope_id, carried_from=body.get("carried_from") or source)
                        db.execute("INSERT OR REPLACE INTO entries VALUES (?,?)", (body["id"], encoded(body)))
                publish(config, db)
            finally:
                db.close()
        return [b["id"] for b in keep]
