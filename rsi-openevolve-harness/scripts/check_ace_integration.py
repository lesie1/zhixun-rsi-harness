"""Real A training feedback + fixture ACE replies; B/C interface fixtures only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from team_ace.__main__ import HARNESS, bounded_worker, make_config
from team_ace.store import BASE, atomic_text, connect, digest, encoded, flush, initialize, location, status


def main():
    summary_path = BASE / 'checks/integration.json'
    if summary_path.exists():
        raise FileExistsError('integration evidence already exists; do not overwrite it')
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    sys.path.insert(0, str(HARNESS))
    from harness.evaluator import evaluate_policy
    from harness.records import on_iteration
    from harness.prompt_extras import get_extras
    from harness.runcfg import load_run_config
    from harness.stage import StageContext

    summaries = []
    for task in 'ABC':
        cfg = make_config(task, 'smoke', 'fixture')
        cfg['ace']['record_origin'] = 'real_A_fixed_continuation' if task == 'A' else 'synthetic_interface_fixture'
        initialize(cfg)
        path = BASE / 'configs' / ('smoke_' + task + '.json')
        atomic_text(path, json.dumps(cfg, ensure_ascii=False, indent=2) + '\n')
        os.environ['HARNESS_RUN_CONFIG'] = str(path)
        os.environ['HARNESS_RUN_ID'] = 'ace-interface-smoke-' + task
        before_prompt = get_extras()['playbook']
        actual_training = None
        if task == 'A':
            context = StageContext(load_run_config(path))
            initial = HARNESS / 'harness/policy_initial.py'
            evaluated = evaluate_policy(initial, context)
            assert evaluated['report']['status'] == 'ok'
            initial_code = initial.read_text()
            # No fabricated LLM proposal: this is the fixed initial policy.
            event = {'kind': 'evaluated', 'iteration': 0, 'parent_id': 'fixed-reference',
                'parent_program': {'id': 'fixed-reference', 'code': initial_code, 'metrics': evaluated['metrics']},
                'child_program': {'id': 'fixed-check', 'code': initial_code, 'metrics': evaluated['metrics']},
                'llm_response': 'HYPOTHESIS: 固定初始策略接口验证；没有 LLM 提案。\nUSED_BULLETS: []',
                'prompt': {'user': before_prompt}, 'token_usage': None, 'is_new_best': False}
            on_iteration(event)  # Actual teammate hook dynamically imports team_ace.on_batch.
            actual_training = {'candidate_metrics': evaluated['report']['metrics']['candidate'],
                'split': context.manifest['counts'], 'flags': evaluated['report']['flags'],
                'fixed_training_executions': 2,
                'explanation': 'stage reference + identical initial policy; both resume 5 epochs; not Agent optimization'}
        else:
            # No B/C training. Exercise the failure-event path without inventing scores.
            on_iteration({'kind': 'failed', 'iteration': 0, 'error': 'Synthetic interface fixture; no training run.',
                'llm_response': 'HYPOTHESIS: 模拟失败输入，只测试任务隔离。\nUSED_BULLETS: []',
                'prompt': {'user': before_prompt}})
        assert status(cfg)['records'] == 1
        assert flush(cfg) == 0
        worker_result = bounded_worker(path, seconds=25)
        assert worker_result['exit_code'] == 0 and worker_result['stop_reason'] is None
        assert status(cfg)['jobs'] == {'done': 1}
        after_prompt = get_extras()['playbook']
        db = connect(cfg)
        try:
            entries = [json.loads(row['body']) for row in db.execute('SELECT body FROM entries')]
            assert len(entries) == 1 and entries[0]['task'] == task
            assert entries[0]['id'] in after_prompt and entries[0]['id'] not in before_prompt
            assert entries[0]['status'] == 'candidate' and entries[0]['mode'] == 'fixture'
            assert db.execute('SELECT count(*) FROM calls').fetchone()[0] == 2
        finally:
            db.close()
        text = Path(cfg['playbook_path']).read_text()
        version = hashlib.sha256(text.strip().encode()).hexdigest()[:12]
        assert 'PLAYBOOK_VERSION: ' + version in after_prompt
        atomic_text(Path(cfg['work_dir']) / 'prompt-before.txt', before_prompt)
        atomic_text(Path(cfg['work_dir']) / 'prompt-after.txt', after_prompt)
        # A clean live library must remain empty after all fixture processing.
        live_path = BASE / 'configs' / ('run_' + task + '.json')
        live_cfg = json.loads(live_path.read_text())
        assert status(live_cfg)['entries'] == 0
        summaries.append({'task': task, 'real_training': actual_training,
            'scope': cfg['ace']['scope'], 'state': status(cfg), 'prompt_version': version,
            'entry_injected_next_prompt': True, 'replay_added_duplicates': False,
            'live_library_entries': status(live_cfg)['entries']})
    before = json.loads((BASE / 'checks/ace-merge-protected.json').read_text())
    unchanged = all(Path(path).is_file() and hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
                    for path, expected in before.items())
    assert unchanged
    result = {'status': 'passed', 'real_external_model_calls': 0, 'fixture_ace_role_calls': 6,
        'real_training_tasks': ['A'], 'real_A_fixed_training_executions': 2,
        'real_training_improvement_claimed': False, 'ABC': summaries,
        'protected_files_unchanged': len(before),
        'scope': 'ACE upstream Reflector/Curator integration with simulated replies; no real Qwen or RSI efficacy result'}
    atomic_text(summary_path, json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(encoded({'status': 'passed', 'tasks': list('ABC'), 'protected_files_unchanged': unchanged,
        'real_model_calls': 0, 'fixture_ace_calls': 6, 'real_A_training_executions': 2}))


if __name__ == '__main__':
    main()
