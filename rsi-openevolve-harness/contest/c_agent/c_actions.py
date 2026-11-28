#!/usr/bin/env python3
"""C: checked JSON actions, real CPU evaluation, explicit opt-in live planning."""
import argparse
from dataclasses import asdict
import importlib.util
import json
import os
import sqlite3
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PYTHON = Path(sys.executable)
for tag in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'BASELINE_CPU_THREADS'):
    os.environ[tag] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True

from action_space import DEFAULT, MENU, apply_action
from backend import PROJECT, file_hash, write_json
from controller import Controller, describe, new_run, setup
from feedback import Rules, extra_violations
from planner import BudgetExhausted, FixturePlanner, LivePlanner
from experience import ExperienceStore


def resolve_run(mode, explicit=None):
    run = Path(explicit) if explicit else Path(json.loads((HERE / f'latest-{mode}.json').read_text(encoding='utf-8'))['run_dir'])
    run = run.resolve()
    if not run.is_relative_to(HERE / 'runs'):
        raise ValueError('Run must be inside this C-actions directory')
    return run


def launch(mode, steps, run_config=None, no_experience=False):
    if not 1 <= steps <= (6 if mode == 'offline' else 5):
        raise ValueError('Use 1..6 offline steps or 1..5 live steps')
    run = new_run(HERE, mode)
    manifest = {'mode': mode, 'status': 'running', 'run_dir': str(run), 'steps_requested': steps,
                'real_api_authorization': 'requires explicit user approval before invoking live',
                'max_output_tokens_per_call': 600, 'provider_retries': 0}
    write_json(run / 'manifest.json', manifest)
    write_json(HERE / f'latest-{mode}.json', {'run_dir': str(run)})
    backend = setup(HERE, run, mode, run_config)
    before = backend.protected()
    write_json(run / 'protected-before.json', before)
    config = json.loads((run/'config.json').read_text(encoding='utf-8'))
    memory = config['experience']
    if no_experience:
        memory['enabled'] = False
        write_json(run/'config.json', config)
        protocol = json.loads((run/'protocol.json').read_text(encoding='utf-8'))
        protocol['experience'] = memory
        write_json(run/'protocol.json', protocol)
    store = None
    try:
        store = ExperienceStore(memory['storage_root'], mode, backend.experience_context, asdict(Rules()),
                                enabled=memory['enabled'], limit=memory['retrieve_k'], max_chars=memory['max_prompt_chars'])
    except (OSError, ValueError, sqlite3.Error) as exc:
        manifest['experience_error'] = type(exc).__name__
        print('经验库初始化失败，当前实验继续但不读取/更新经验：'+type(exc).__name__, flush=True)
    controller = Controller(run, backend, experience=store)
    planner = FixturePlanner() if mode == 'offline' else LivePlanner(run, steps)
    print('离线模拟决策，不调用真实API。' if mode == 'offline' else '真实Qwen决策，将消耗API额度。', flush=True)
    print('结果目录：' + str(run), flush=True)
    error = None
    try:
        for turn in range(1, steps + 1):
            try:
                record = controller.step(planner, turn)
            except BudgetExhausted as exc:
                manifest['stop_reason'] = str(exc)
                break
            p = record['proposal'] or {}
            print(f"{record['id']} {p.get('action', 'invalid')} → {record['decision']['reason']}", flush=True)
            write_json(run / 'state.json', controller.snapshot())
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        after = backend.protected()
        changed = [name for name in before if before[name] != after.get(name)]
        state = controller.snapshot()
        write_json(run / 'state.json', state)
        write_json(run / 'protected-after.json', after)
        manifest.update(status='failed' if error or changed else 'completed', error_type=error,
                        protected_files_changed=changed, real_api_calls=0 if mode == 'offline' else 'see provider-ledger.json',
                        steps_recorded=len(state['history']), gate_evaluated=False)
        write_json(run / 'manifest.json', manifest)
        describe(run, state)
    return 0 if manifest['status'] == 'completed' else 1


def benchmark():
    run = new_run(HERE, 'benchmark')
    backend = setup(HERE, run, 'benchmark')
    before = backend.protected()
    controller = Controller(run, backend)
    cases = [('single:' + action, [action]) for action in MENU if action != 'keep']
    cases += [('combo:' + action, ['pick_best_epoch', action]) for action in
              ['lr_half', 'lr_decay', 'high_rul_weight_1_2', 'low_rul_weight_1_2', 'both_tail_weight_1_2',
               'epochs_1', 'epochs_3', 'tail_error_1_1']]
    cases += [('short_tail_half', ['pick_best_epoch', 'epochs_3', 'tail_error_1_1', 'lr_half'])]
    rows = []
    for name, sequence in cases:
        settings = DEFAULT
        for action in sequence:
            settings = apply_action(action, settings)
        candidate, reused = backend.evaluate(settings)
        control_settings = DEFAULT
        for action in sequence[:-1]:
            control_settings = apply_action(action, control_settings)
        control, _ = backend.evaluate(control_settings)
        from feedback import select
        decision = select(candidate.feedback, control.feedback, backend.parent.feedback, Rules())
        rows.append({'case': name, 'settings': asdict(settings), 'feedback': candidate.feedback,
                     'control_settings': asdict(control_settings), 'control_feedback': control.feedback,
                     'decision_vs_control': decision, 'policy': candidate.policy,
                     'checkpoint': candidate.checkpoint, 'reused': reused})
        metric = candidate.feedback.get('metrics', {}).get('rmse')
        print(name, metric, decision['reason'], flush=True)
    after = backend.protected()
    changed = [k for k, value in before.items() if after[k] != value]
    write_json(run / 'protected-before.json', before)
    write_json(run / 'protected-after.json', after)
    report = {'mode': 'benchmark', 'rows': rows, 'parent': backend.parent.record(),
              'reference': backend.reference.record(), 'real_api_calls': 0, 'gate_evaluated': False,
              'protected_files_changed': changed, 'attribution': 'fixed human-designed controls, not Agent-generated improvement'}
    write_json(run / 'benchmark.json', report)
    write_json(HERE / 'latest-benchmark.json', {'run_dir': str(run)})
    print('BENCHMARK_JSON', run / 'benchmark.json')
    return 0 if not changed and all(x['feedback']['status'] == 'ok' for x in rows) else 1


def freeze(run):
    manifest = json.loads((run / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['mode'] != 'live' or manifest['status'] != 'completed':
        raise ValueError('Freeze only a completed live experiment')
    path = run / 'frozen.json'
    if path.exists():
        raise ValueError('This experiment is already frozen')
    state = json.loads((run / 'state.json').read_text(encoding='utf-8'))
    winner = state['incumbent']
    frozen = {'incumbent': winner, 'rules': state['rules'],
              'checkpoint_sha256': file_hash(winner['checkpoint']),
              'policy_sha256': file_hash(winner['policy']) if winner['policy'] else None,
              'config_sha256': file_hash(run / 'config.json'), 'state_sha256': file_hash(run / 'state.json'),
              'protected': json.loads((run / 'protected-after.json').read_text(encoding='utf-8'))}
    if any(file_hash(name) != value for name, value in frozen['protected'].items()):
        raise ValueError('Source data or trainer changed since this experiment')
    write_json(path, frozen)
    print('已冻结；尚未执行独立验收。', path)


def gate_choice(result, compact, rules):
    """Add declared tail safeguards without changing the original gate's evidence."""
    extra = {name: extra_violations(compact[name], compact['parent'], rules)
             for name in ('candidate', 'reference')}
    extra['candidate_vs_reference'] = extra_violations(compact['candidate'], compact['reference'], rules)
    decision, reason = result['decision'], result['reason']
    safe_reference = result['comparisons']['reference_vs_parent']['passes'] and not extra['reference']
    if decision == 'adopt_candidate' and (extra['candidate'] or
                                         (safe_reference and extra['candidate_vs_reference'])):
        decision = 'adopt_reference' if safe_reference else 'keep_parent'
        reason = '候选触发本地额外护栏；' + ('回退到通过验收的官方续训' if safe_reference else '保留父模型')
    elif decision == 'adopt_reference' and extra['reference']:
        decision, reason = 'keep_parent', '官方续训触发本地额外护栏；保留父模型'
    return decision, reason, extra


def gate(run):
    frozen = json.loads((run / 'frozen.json').read_text(encoding='utf-8'))
    if (run / 'gate-started.json').exists():
        raise ValueError('Gate already started; do not repeatedly tune on gate feedback')
    if (file_hash(run / 'config.json') != frozen['config_sha256'] or
            file_hash(run / 'state.json') != frozen['state_sha256'] or
            file_hash(frozen['incumbent']['checkpoint']) != frozen['checkpoint_sha256']):
        raise ValueError('Frozen experiment changed')
    if frozen['incumbent']['policy'] and file_hash(frozen['incumbent']['policy']) != frozen['policy_sha256']:
        raise ValueError('Frozen policy changed')
    if any(file_hash(name) != value for name, value in frozen['protected'].items()):
        raise ValueError('Frozen source data or trainer changed')
    # Explicit command only. Importing/benchmarking/offline planning never opens gate metrics.
    backend = setup_gate(run)
    from baselines.model import Model
    from harness.gate import decide
    from feedback import compact_report
    candidate = Model.load(frozen['incumbent']['checkpoint'])
    write_json(run / 'gate-started.json', {'frozen_sha256': file_hash(run / 'frozen.json')})
    result = decide(backend, candidate, resamples=2000, seed=42)
    data, meta = backend.load_gate()
    from harness.groups import discover_groups
    info = discover_groups(meta)
    models = {'parent': backend.parent, 'reference': backend.reference_model, 'candidate': candidate}
    compact = {}
    for name, model in models.items():
        bundle = backend.analyze(model, data, meta, info)
        compact[name] = compact_report({'status': 'ok', 'metrics': {'candidate': result['metrics'][name]},
                                       'slices': {'by_quantile': bundle['by_quantile'], 'by_group': bundle['by_group']}})
    rules = Rules(**frozen['rules'])
    decision, reason, extra = gate_choice(result, compact, rules)
    result.update(core_decision=result['decision'], core_reason=result['reason'],
                  decision=decision, reason=reason, local_tail_violations=extra,
                  selected_checkpoint={'adopt_candidate': frozen['incumbent']['checkpoint'],
                                       'adopt_reference': str(backend.reference_dir / 'checkpoint.npz'),
                                       'keep_parent': backend.parent_path}[decision],
                  interpretation='local independent acceptance, not platform hidden score')
    write_json(run / 'gate-decision.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def setup_gate(run):
    sys.path.insert(0, str(PROJECT))
    from harness.runcfg import load_run_config
    from harness.stage import get_context
    return get_context(load_run_config(run / 'config.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('check')
    sub.add_parser('benchmark')
    skill_parser = sub.add_parser('skills', help='Inspect persistent evidence cards; no training or API calls')
    skill_parser.add_argument('--mode', choices=('live', 'offline'), default='live')
    skill_parser.add_argument('--import-run', help='Import one completed compatible real-Qwen run, with provenance/metric verification')
    for name in ('offline', 'live'):
        p = sub.add_parser(name)
        p.add_argument('--steps', type=int, default=6 if name == 'offline' else 1)
        p.add_argument('--run-config', help='Absolute per-round harness config; supplies the current parent and public data paths')
        p.add_argument('--no-experience', action='store_true', help='Do not retrieve or learn cards in this experiment')
    for name in ('status', 'freeze', 'gate'):
        p = sub.add_parser(name)
        p.add_argument('--run')
        if name == 'status':
            p.add_argument('--mode', choices=('offline', 'live'), default='offline')
    args = parser.parse_args()
    if args.command == 'skills':
        config = json.loads((HERE/'base_C.json').read_text(encoding='utf-8'))
        memory = config.get('experience') or {}
        memory_root = Path(memory.get('storage_root') or HERE/'experience')
        if not memory_root.is_absolute():
            memory_root = HERE/memory_root
        if args.import_run:
            if args.mode != 'live' or memory.get('enabled', True) is not True:
                raise ValueError('Import requires enabled live experience memory')
            from experience_import import import_verified_run
            result = import_verified_run(resolve_run('live', args.import_run),
                                         memory_root.resolve(), HERE/'initial_policy.py',
                                         memory.get('retrieve_k', 3), memory.get('max_prompt_chars', 2600))
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        store = ExperienceStore(memory_root.resolve(), args.mode,
                                enabled=memory.get('enabled', True))
        print(json.dumps({**store.summary(), 'recent_cards': [store.brief(c) for c in store.cards(all_domains=True)[-3:]]},
                         ensure_ascii=False, indent=2))
        return 0
    if args.command == 'check':
        config = json.loads((HERE / 'base_C.json').read_text(encoding='utf-8'))
        for name in ('train_path', 'validation_path', 'parent_checkpoint', 'split_manifest'):
            if not Path(config[name]).is_file():
                raise ValueError('Missing input: ' + name)
        sys.path.insert(0, str(PROJECT))
        from harness.constraints import check_policy_source
        from action_space import render_policy
        initial = (HERE / 'initial_policy.py').read_text(encoding='utf-8')
        for action in MENU:
            settings = apply_action(action, DEFAULT)
            if settings is not None and check_policy_source(render_policy(initial, settings)):
                raise ValueError('Policy constraint failure: ' + action)
        print(json.dumps(MENU, ensure_ascii=False, indent=2))
        print('数据路径、模块导入、全部动作静态约束通过；未训练、未调用API、未验收。')
        return 0
    if args.command == 'benchmark':
        return benchmark()
    if args.command in ('offline', 'live'):
        return launch(args.command, args.steps, args.run_config, args.no_experience)
    run = resolve_run(args.mode if args.command == 'status' else 'live', args.run)
    if args.command == 'status':
        describe(run, json.loads((run / 'state.json').read_text(encoding='utf-8')))
    elif args.command == 'freeze':
        freeze(run)
    else:
        gate(run)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        print('配置/实验错误：' + str(error), file=sys.stderr)
        sys.exit(2)
