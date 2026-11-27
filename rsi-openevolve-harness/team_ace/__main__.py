"""CLI for setup, recovery and bounded asynchronous ACE processing."""
import argparse
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .store import BASE, TASKS, atomic_text, connect, digest, encoded, exclusive, flush, initialize, location, publish, rollback, scope_for, status

HARNESS = BASE
DATA = Path('/home/dino/Documents/cpu-baselines-20261110-v1')


def make_config(task, namespace='live', mode='disabled'):
    if task not in TASKS or namespace not in ('live', 'smoke'):
        raise ValueError('unsupported task or namespace')
    config = {'task_name': task, 'stage_id': 'ace-' + namespace + '-' + task,
        'train_path': str(DATA / task / 'public/train.jsonl'),
        'validation_path': str(DATA / task / 'public/validation.jsonl'),
        'parent_checkpoint': str(DATA / task / 'checkpoint.npz'),
        'work_dir': str(BASE / 'runs' / namespace / task), 'seed': 42,
        'limits': {'max_epochs': 5, 'train_timeout_s': 20, 'cpu_threads': 1},
        'split': {'min_groups': 4}, 'records': {'include_prompt': True},
        'prompt': {'playbook_max_chars': 2500, 'recent_failures': 3},
        'ace': {'module': 'team_ace', 'batch_size': 1, 'mode': mode,
                'storage_root': str(BASE / 'state' / namespace),
                'record_origin': 'live_training' if namespace == 'live' else 'mixed_smoke_see_record',
                'max_total_calls': 0 if mode == 'disabled' else 6,
                'live_calls_approved': False, 'model': None, 'api_base': None, 'api_key_env': None}}
    config['ace']['scope'] = scope_for(config)
    folder = Path(config['ace']['storage_root']) / task / digest(config['ace']['scope'])[:16]
    config['playbook_path'] = str(folder / 'playbook.md')
    return config


def bounded_worker(config_path, seconds=35):
    config = json.loads(Path(config_path).read_text())
    folder = location(config)
    folder.mkdir(parents=True, exist_ok=True)
    log = folder / ('worker-' + str(time.time_ns()))
    command = [sys.executable, '-B', '-m', 'team_ace.worker', '--config', str(Path(config_path).resolve())]
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'LANG', 'LC_ALL', 'LD_LIBRARY_PATH')}
    # Pass only the designated credential, never the entire parent environment.
    key = config['ace'].get('api_key_env')
    if config['ace'].get('mode') == 'live' and key and key in os.environ:
        env[key] = os.environ[key]
    env.update(PYTHONPATH=str(BASE), PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1',
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with log.with_suffix('.stdout').open('x') as stdout, log.with_suffix('.stderr').open('x') as stderr:
        p = subprocess.Popen(command, cwd=BASE, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        started = time.monotonic()
        reason = None
        while p.poll() is None:
            if time.monotonic() - started > seconds:
                reason = 'wall_timeout'
            try:
                memory = Path(f'/proc/{p.pid}/status').read_text()
                rss = int(next(line.split()[1] for line in memory.splitlines() if line.startswith('VmRSS:')))
                available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                     if line.startswith('MemAvailable:')))
                if rss > 384 * 1024 or available < 250 * 1024:
                    reason = 'memory_limit'
            except (FileNotFoundError, StopIteration, ProcessLookupError):
                pass
            if reason:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait()
                break
            time.sleep(0.05)
    result = {'exit_code': p.returncode, 'stop_reason': reason, 'log': str(log),
              'seconds': time.monotonic() - started}
    atomic_text(log.with_suffix('.json'), encoded(result) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['init', 'status', 'flush', 'process', 'serve', 'retry', 'retire', 'render', 'rollback'])
    parser.add_argument('--config')
    parser.add_argument('--max-jobs', type=int, default=1)
    parser.add_argument('--seconds', type=int, default=45)
    parser.add_argument('--entry-id')
    parser.add_argument('--version')
    args = parser.parse_args()
    if not 1 <= args.max_jobs <= 3 or not 1 <= args.seconds <= 120:
        parser.error('max-jobs must be 1..3 and seconds 1..120')
    if args.command == 'init':
        for task in TASKS:
            path = BASE / 'configs' / ('run_' + task + '.json')
            if path.exists():
                raise FileExistsError('configuration already exists; will not overwrite ' + str(path))
            config = make_config(task)
            initialize(config)
            atomic_text(path, json.dumps(config, ensure_ascii=False, indent=2) + '\n')
        print('Initialized A/B/C separately; live calls disabled.')
        return
    if not args.config:
        parser.error('--config required')
    config = json.loads(Path(args.config).read_text())
    if args.command == 'flush':
        print(encoded({'new_records': flush(config)}))
    elif args.command in ('process', 'serve'):
        deadline = time.monotonic() + args.seconds
        results = []
        while len(results) < args.max_jobs and time.monotonic() < deadline:
            if args.command == 'serve':
                flush(config)  # Also recover the harness tail and previously missed callbacks.
            info = status(config)
            if info['jobs'].get('pending', 0) and config['ace'].get('mode') != 'disabled':
                result = bounded_worker(args.config, min(35, deadline - time.monotonic()))
                results.append(result)
                if result['exit_code'] != 0:
                    break
            elif args.command == 'process' or config['ace'].get('mode') == 'disabled':
                break
            else:
                time.sleep(0.2)
        print(encoded({'processing': results, 'state': status(config)}))
    elif args.command == 'rollback':
        print(encoded({'version': rollback(config, args.version)}))
    elif args.command in ('retry', 'retire', 'render'):
        with exclusive(config):
            db = connect(config)
            try:
                with db:
                    if args.command == 'retry':
                        db.execute("UPDATE jobs SET status='pending' WHERE status='failed' AND attempts<2")
                    elif args.command == 'retire':
                        row = db.execute('SELECT body FROM entries WHERE id=?', (args.entry_id,)).fetchone()
                        if row is None:
                            raise ValueError('unknown entry')
                        value = json.loads(row['body'])
                        value['status'] = 'retired'
                        db.execute('UPDATE entries SET body=? WHERE id=?', (encoded(value), args.entry_id))
                print(encoded({'version': publish(config, db)}))
            finally:
                db.close()
    else:
        print(encoded(status(config)))


if __name__ == '__main__':
    main()
