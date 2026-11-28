"""Import compatible, completed real-Qwen selection experiments without replaying API/training."""
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

from action_space import DEFAULT, render_policy, validate_settings
from backend import PROJECT, file_hash
from experience import ExperienceStore, memory_context
from feedback import Rules, parse_proposal


def complete_settings(value):
    if value is None:
        return None
    defaults = asdict(DEFAULT)
    if not isinstance(value, dict) or not set(value) <= set(defaults):
        raise ValueError('Unknown historical policy settings')
    defaults.update(value)
    return asdict(validate_settings(defaults))


def valid_live_proof(call, ledger, trial, observation):
    if (call.get('mode') != 'live' or call.get('status') != 'ok' or call.get('transport_simulated') is not False or
            call.get('finish_reason') != 'stop' or call.get('returned_model') != 'Qwen/Qwen3.8-27B' or
            not str(call.get('provider_id', '')).startswith('chatcmpl-') or ledger.get('status') != 'ok' or
            trial.get('simulated_planner') is not False or trial.get('provider', {}).get('simulated') is not False):
        raise ValueError('Missing real-provider provenance')
    usage = call.get('usage') or {}
    if type(usage.get('total_tokens')) is not int or usage['total_tokens'] <= 0:
        raise ValueError('Missing provider token usage')
    proposal = parse_proposal(call.get('raw'), observation['available_evidence_ids'])
    if proposal != trial.get('proposal'):
        raise ValueError('Provider reply and recorded action differ')


def import_verified_run(run, root, initial_path, limit=3, max_chars=2600):
    run = Path(run).resolve()
    def read(name):
        return json.loads((run/name).read_text(encoding='utf-8'))
    manifest, state, config = read('manifest.json'), read('state.json'), read('config.json')
    if (config.get('task_name') != 'C' or manifest.get('mode') != 'live' or manifest.get('status') != 'completed' or
            not state.get('selection_only') or state.get('gate_evaluated') or
            (run/'gate-started.json').exists()):
        raise ValueError('Only completed live selection experiments without gate access can be imported')
    for filename in ('protected-before.json', 'protected-after.json'):
        if any(file_hash(path) != digest for path, digest in read(filename).items()):
            raise ValueError('Historical protected source/data changed')
    sys.path.insert(0, str(PROJECT))
    from baselines.model import Model
    from harness.dataio import load_dataset
    from harness.metrics import regression_metrics
    from harness.constraints import check_checkpoint
    parent = Model.load(config['parent_checkpoint'])
    selection = load_dataset(run/'work/splits/select.jsonl', parent.meta['task'], parent.meta['classes'],
                             parent.meta['paths'], cache_dir=None)
    context = memory_context(parent, file_hash(config['train_path']), selection,
                             file_hash(initial_path), file_hash(Path(__file__).with_name('action_space.py')))
    if state.get('rules') != asdict(Rules()):
        raise ValueError('Historical selection rules differ')
    store = ExperienceStore(root, 'live', context, asdict(Rules()), limit=limit, max_chars=max_chars)
    initial = Path(initial_path).read_text(encoding='utf-8')
    before_record = state['reference'] if state['reference_decision']['accepted'] else state['parent']
    before, before_settings = before_record['feedback'], complete_settings(before_record.get('settings'))
    original_control = Model.load(before_record['checkpoint'])
    control_metrics = regression_metrics(selection['y'], original_control.predict(selection['x']), with_nasa=True)
    if any(not math.isclose(control_metrics[key], before['metrics'][key], rel_tol=1e-10, abs_tol=1e-8)
           for key in ('rmse', 'mae', 'nasa_score')):
        raise ValueError('Historical control metrics do not match its checkpoint')
    ledger = read('provider-ledger.json')
    outcomes = []
    for filename in sorted((run/'trials').glob('*.json')):
        number = int(filename.stem)
        trial = read(str(filename.relative_to(run)))
        if not trial.get('new_training') or trial.get('reused'):
            outcomes.append({'trial': trial['id'], 'status': 'skipped_no_new_measurement'})
            continue
        valid_live_proof(read(f'calls/{number:03d}.json'), ledger[str(number)], trial,
                         read(f'observations/{number:03d}.json'))
        trial['base_settings'] = complete_settings(trial['base_settings'])
        trial['settings'] = complete_settings(trial['settings'])
        policy = Path(trial['policy']).resolve()
        checkpoint = Path(trial['checkpoint']).resolve()
        if not policy.is_relative_to(run/'work') or not checkpoint.is_relative_to(run/'work'):
            raise ValueError('Historical artifact escaped the run directory')
        if policy.read_text(encoding='utf-8') != render_policy(initial, validate_settings(trial['settings'])):
            raise ValueError('Historical policy differs from current bounded implementation')
        model, problems = check_checkpoint(parent, checkpoint)
        if problems:
            raise ValueError('Historical candidate violates continuation constraints')
        metrics = regression_metrics(selection['y'], model.predict(selection['x']), with_nasa=True)
        if any(not math.isclose(metrics[key], trial['candidate']['metrics'][key], rel_tol=1e-10, abs_tol=1e-8)
               for key in ('rmse', 'mae', 'nasa_score')):
            raise ValueError('Historical candidate metrics do not match its checkpoint')
        result = store.learn(trial, before, before_settings,
                             {'run': run.name, 'trial': trial['id'], 'trial_sha256': file_hash(filename),
                              'imported': True, 'verification': 'real_provider_policy_and_selection_metric_check'})
        outcomes.append({'trial': trial['id'], **result})
        if trial['decision']['accepted']:
            before, before_settings = trial['candidate'], trial['settings']
    return {'source_run': str(run), 'verification': 'no API/training replay; selection metrics recomputed',
            'outcomes': outcomes, 'library': store.summary()}
