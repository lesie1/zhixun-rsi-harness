"""Finite-action experiment controller; selection feedback never includes gate scores."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import sqlite3
from pathlib import Path
import uuid

from action_space import DEFAULT, MENU, Settings, apply_action
from backend import HarnessBackend, write_json
from feedback import Rules, observation, parse_proposal, select
from planner import BudgetExhausted, messages_for


def new_run(home, mode):
    name = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    run = Path(home) / 'runs' / mode / name
    run.mkdir(parents=True, exist_ok=False)
    return run


def history_view(record):
    return {k: record.get(k) for k in ('id', 'proposal', 'base_settings', 'settings', 'candidate', 'decision', 'reused', 'new_training', 'experience_used')}


class Controller:
    def __init__(self, run, backend, rules=None, experience=None):
        self.run = Path(run)
        self.backend = backend
        self.rules = rules or Rules()
        self.parent = backend.parent
        self.reference = backend.reference
        self.incumbent = self.parent
        self.history = []
        self.experience = experience
        self.experience_errors = []
        self.reference_decision = select(self.reference.feedback, self.parent.feedback,
                                         self.parent.feedback, self.rules)
        if self.reference_decision['accepted']:
            self.incumbent = self.reference

    def observe(self):
        result = observation(self.parent.feedback, self.reference.feedback, self.incumbent.feedback,
                             self.incumbent.settings or DEFAULT, self.history, self.rules)
        if self.experience is not None:
            cards = []
            if not self.experience_errors:
                try:
                    cards = self.experience.retrieve(self.incumbent.feedback, self.incumbent.settings or DEFAULT)
                except (OSError, ValueError, sqlite3.Error) as error:
                    self.experience_errors.append('retrieval:'+type(error).__name__)
            result['experience'] = {'enabled': self.experience.enabled, 'namespace': self.experience.namespace,
                                    'cards': cards, 'status': 'degraded' if self.experience_errors else 'ok',
                                    'use': 'prior public-selection evidence; never a replacement for current evaluation'}
            result['available_evidence_ids'] += [card['id'] for card in cards]
        return result

    def step(self, planner, turn):
        obs = self.observe()
        write_json(self.run / 'observations' / f'{turn:03d}.json', obs)
        write_json(self.run / 'experience-retrieved' / f'{turn:03d}.json', obs.get('experience', {'enabled': False}))
        before, before_settings = self.incumbent.feedback, self.incumbent.settings
        base = self.incumbent.settings or DEFAULT
        record = {'id': f'T{turn:04d}', 'proposal': None, 'base_settings': asdict(base),
                  'candidate': None, 'decision': None, 'reused': False, 'new_training': False,
                  'simulated_planner': planner.simulated}
        try:
            raw, metadata = planner.propose(obs, turn)
            record['provider'] = metadata
            (self.run / 'replies').mkdir(exist_ok=True)
            (self.run / 'replies' / f'{turn:03d}.txt').write_text(raw, encoding='utf-8')
            proposal = parse_proposal(raw, obs['available_evidence_ids'])
            record['proposal'] = proposal
            record['experience_used'] = [item for item in proposal['used'] if item.startswith('CEXP-')]
            settings = apply_action(proposal['action'], base)
            if settings is None:
                record.update(candidate=self.parent.feedback, reused=True)
                record['decision'] = {'accepted': False, 'reason': 'keep_parent_option_no_new_training',
                                      'parent_checkpoint': self.parent.checkpoint, 'violations': []}
            else:
                candidate, reused = self.backend.evaluate(settings)
                decision = select(candidate.feedback, self.incumbent.feedback, self.parent.feedback, self.rules)
                record.update(candidate=candidate.feedback, reused=reused, new_training=not reused,
                              settings=asdict(settings), policy=candidate.policy, checkpoint=candidate.checkpoint,
                              decision=decision)
                if decision['accepted']:
                    self.incumbent = candidate
        except BudgetExhausted:
            raise
        except (ValueError, TypeError, KeyError, RuntimeError) as error:
            record['decision'] = {'accepted': False, 'reason': 'invalid_reply_or_execution_failure',
                                  'error': str(error)[:400], 'violations': []}
        self.history.append(history_view(record))
        write_json(self.run / 'trials' / f'{turn:03d}.json', record)
        if self.experience is not None and not self.experience_errors:
            try:
                from backend import file_hash
                update = self.experience.learn(record, before, before_settings,
                                               {'run': self.run.name, 'trial': record['id'],
                                                'trial_sha256': file_hash(self.run/'trials'/f'{turn:03d}.json')})
            except (OSError, ValueError, sqlite3.Error) as error:
                self.experience_errors.append('learning:'+type(error).__name__)
                update = {'status': 'degraded', 'error_type': type(error).__name__}
            write_json(self.run / 'experience-updates' / f'{turn:03d}.json', update)
        return record

    def snapshot(self):
        return {'parent': self.parent.record(), 'reference': self.reference.record(),
                'reference_decision': self.reference_decision, 'incumbent': self.incumbent.record(),
                'history': self.history, 'rules': asdict(self.rules),
                'selection_only': True, 'gate_evaluated': False,
                'experience': self.experience_state(),
                'unique_policy_evaluations': self.backend.calls}

    def experience_state(self):
        if self.experience is None:
            return {'enabled': False}
        return {'enabled': self.experience.enabled, 'namespace': self.experience.namespace,
                'new_events_this_run': self.experience.inserted,
                'duplicates_ignored': self.experience.duplicates, 'errors': self.experience_errors,
                'database': str(self.experience.path)}


def describe(run, state):
    lines = ['C 有限动作实验（仅挑选集，未执行最终验收）',
             f"父模型 RMSE：{state['parent']['feedback']['metrics']['rmse']:.6f}",
             f"官方续训 RMSE：{state['reference']['feedback']['metrics']['rmse']:.6f}",
             f"当前搜索最佳 RMSE：{state['incumbent']['feedback']['metrics']['rmse']:.6f}"]
    memory = state.get('experience', {})
    if memory.get('enabled'):
        lines.append(f"经验库：{memory['namespace']}；本次新增证据 {memory['new_events_this_run']}；去重 {memory['duplicates_ignored']}；异常 {memory['errors']}")
    for trial in state['history']:
        proposal = trial['proposal'] or {}
        value = trial['candidate']
        rmse = value.get('metrics', {}).get('rmse') if value else None
        lines += ['', trial['id'] + ' 动作：' + proposal.get('action', '无效回复'),
                  '依据：' + proposal.get('why', '未产生有效动作'),
                  '结果：' + (f'RMSE={rmse:.6f}' if isinstance(rmse, (int, float)) else '无有效指标'),
                  '选择：' + trial['decision']['reason'],
                  '护栏：' + json.dumps(trial['decision'].get('violations', []), ensure_ascii=False),
                  '新训练：' + str(trial['new_training']) + '；复用已有测量：' + str(trial['reused'])]
    lines += ['', '没有把挑选集选择当作平台采纳或独立验收。', '结果目录：' + str(run)]
    text = '\n'.join(lines) + '\n'
    (Path(run) / 'review.txt').write_text(text, encoding='utf-8')
    print(text)


def setup(home, run, mode, base_config=None):
    home, run = Path(home), Path(run)
    source = Path(base_config).resolve() if base_config else home / 'base_C.json'
    config = json.loads(source.read_text(encoding='utf-8'))
    config.update(stage_id='C-actions-' + run.name, work_dir=str(run / 'work'),
                  playbook_path=str(run / 'unused-playbook.md'))
    config['ace'].update(mode='disabled', batch_size=0, max_total_calls=0,
                         live_calls_approved=False, storage_root=str(run / 'ace-disabled'))
    memory = config.get('experience') or {}
    storage = Path(memory.get('storage_root') or home/'experience')
    if not storage.is_absolute():
        storage = source.parent/storage
    config['experience'] = {'enabled': memory.get('enabled', True),
                            'storage_root': str(storage.resolve()),
                            'retrieve_k': memory.get('retrieve_k', 3), 'max_prompt_chars': memory.get('max_prompt_chars', 2600)}
    write_json(run / 'config.json', config)
    backend = HarnessBackend(run / 'config.json', home / 'initial_policy.py')
    write_json(run / 'baselines.json', backend.export_state())
    write_json(run / 'protocol.json', {
        'revision': 'C-experience-v3-20261127',
        'experience': config['experience'],
        'base_config_source': str(source),
        'mode': mode, 'actions': MENU, 'rules': asdict(Rules()),
        'weight_source': 'training targets only; no target cap; mean-one weights',
        'composition': 'patch incumbent settings; all weights resume from fixed stage parent',
        'keep': 'no new training; parent preserved; never discard measured search best',
        'tail_loss': '1.1 only on low-target overprediction/high-target underprediction; quartiles from each training batch',
        'interpretation': 'fixed-data policy experiment, not official ten-round evaluation',
    })
    return backend
