"""Adapter to the unchanged competition-development CPU harness."""
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import sys

from action_space import DEFAULT, Settings, render_policy, settings_id
from feedback import compact_report
from experience import memory_context


PROJECT = Path(os.environ.get('C_HARNESS_ROOT', str(Path(__file__).resolve().parents[1] / 'runtime'))).resolve()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def evaluation_scope(parent_digest, train_path, select_path, select_ids):
    identity = {'parent_checkpoint_sha256': parent_digest,
                'select_ids': [str(item) for item in select_ids],
                'train_sha256': file_hash(train_path), 'select_sha256': file_hash(select_path)}
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode('utf-8')).hexdigest()


@dataclass
class Candidate:
    id: str
    settings: Settings | None
    feedback: dict
    checkpoint: str | None
    policy: str | None

    def record(self):
        return {'id': self.id, 'settings': asdict(self.settings) if self.settings else None,
                'feedback': self.feedback, 'checkpoint': self.checkpoint, 'policy': self.policy}


class HarnessBackend:
    def __init__(self, config_path, initial_path):
        for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BASELINE_CPU_THREADS'):
            os.environ[name] = '1'
        os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(PROJECT))
        from harness.runcfg import load_run_config
        from harness.stage import get_context
        self.context = get_context(load_run_config(config_path))
        def forbid_gate():
            raise RuntimeError('Final-gate measurements are forbidden during development/search')
        self.context.load_gate = forbid_gate
        self.initial = Path(initial_path).read_text(encoding='utf-8')
        self.work = self.context.work
        self.calls = 0
        self.cache = {}
        self.scope = evaluation_scope(self.context.parent_digest, self.context.config['train_path'],
                                      self.context.select_path, self.context.select['ids'])
        bundle = self.context.parent_bundle
        parent_report = {
            'status': 'ok', 'metrics': {'candidate': self.context.parent_metrics},
            'slices': {'by_quantile': bundle['by_quantile'], 'by_group': bundle['by_group'], 'nasa': bundle.get('nasa', {})},
            'score': {'guard_ok': True, 'violations': []},
        }
        self.parent = Candidate('parent', None, compact_report(parent_report),
                                self.context.parent_path, None)
        self.parent.feedback['evaluation_scope'] = self.scope
        # Reuse the official reference already trained by StageContext. Do not train it twice.
        from harness.feedback import build_report
        runner_report = json.loads((self.context.reference_dir / 'runner_report.json').read_text(encoding='utf-8'))
        reference_report = build_report(self.context.feedback_context(), 'ok',
                                        bundle=self.context.reference_bundle, scoring=self.context.reference_score,
                                        runner_report=runner_report, candidate_metrics=self.context.reference_metrics)
        self.reference = Candidate('reference', DEFAULT, compact_report(reference_report),
                                   str(self.context.reference_dir / 'checkpoint.npz'),
                                   str(Path(initial_path).resolve()))
        self.reference.feedback['evaluation_scope'] = self.scope
        self.cache[settings_id(DEFAULT)] = self.reference
        self.experience_context = memory_context(self.context.parent, file_hash(self.context.config['train_path']),
                                                 self.context.select, file_hash(initial_path),
                                                 file_hash(Path(__file__).with_name('action_space.py')))

    def evaluate(self, settings):
        from harness.evaluator import evaluate_policy
        from harness.constraints import check_policy_source
        identifier = settings_id(settings)
        if identifier in self.cache:
            return self.cache[identifier], True
        policy = self.work / 'action-policies' / (identifier + '.py')
        policy.parent.mkdir(exist_ok=True)
        code = render_policy(self.initial, settings)
        problems = check_policy_source(code)
        if problems:
            raise ValueError('Rendered policy failed fixed constraints: ' + '; '.join(problems))
        if policy.exists() and policy.read_text(encoding='utf-8') != code:
            raise ValueError('Existing policy differs')
        policy.write_text(code, encoding='utf-8')
        result = evaluate_policy(policy, self.context)
        self.calls += 1
        report = result['report']
        candidate_dir = report.get('candidate_dir')
        checkpoint = str(Path(candidate_dir) / 'checkpoint.npz') if candidate_dir else None
        candidate = Candidate('policy-' + identifier, settings, compact_report(report), checkpoint, str(policy))
        candidate.feedback['evaluation_scope'] = self.scope
        self.cache[identifier] = candidate
        return candidate, False

    def protected(self):
        config = self.context.config
        paths = [Path(config[k]) for k in ('parent_checkpoint', 'train_path', 'validation_path', 'split_manifest')]
        paths += [PROJECT / 'harness' / n for n in ['runner.py', 'evaluator.py', 'policy_initial.py', 'metrics.py', 'gate.py']]
        paths += [PROJECT / 'configs/run_C.split-v1.json']
        return {str(p): file_hash(p) for p in paths}

    def export_state(self):
        return {'parent': self.parent.record(), 'reference': self.reference.record(),
                'unique_policy_evaluations': self.calls,
                'parent_checkpoint_sha256': self.context.parent_digest,
                'evaluation_scope': self.scope,
                'experience_context': self.experience_context,
                'train_quantiles': self.context.train_stats['target_quartiles'],
                'select_rows': len(self.context.select['ids']),
                'gate_evaluated': False}
