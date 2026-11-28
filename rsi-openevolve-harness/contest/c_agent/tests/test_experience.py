import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from action_space import DEFAULT, apply_action
from controller import Controller
from experience import ExperienceStore, memory_context
from feedback import Rules
from experience_import import complete_settings, valid_live_proof
from test_actions import FakeBackend, FixedPlanner, observed
import numpy as np

TEMP = ROOT/'.test-tmp'
TEMP.mkdir(exist_ok=True)
CONTEXT = {'domain': {'task': 'C', 'unit': 'cycles', 'policy_version': 'unit-test'}, 'context_id': 'context-one'}


def record(simulated=True, accepted=True):
    return {'id': 'T0001', 'simulated_planner': simulated, 'provider': {'simulated': simulated},
            'proposal': {'action': 'pick_best_epoch', 'why': 'MODEL PROSE MUST NOT BECOME A SKILL', 'used': []},
            'base_settings': asdict(DEFAULT), 'settings': asdict(apply_action('pick_best_epoch', DEFAULT)),
            'new_training': True, 'reused': False, 'candidate': observed(rmse=48),
            'decision': {'accepted': accepted, 'reason': 'rmse_improved' if accepted else 'guard_rejected',
                         'violations': [] if accepted else ['parent:high_underestimation']}}


class MemoryTests(unittest.TestCase):
    def store(self, root, namespace='offline', context=None, **kwargs):
        return ExperienceStore(root, namespace, context or CONTEXT, asdict(Rules()), **kwargs)

    def test_learn_survives_reopen_and_retrieval_exposes_verifiable_card_id(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            first = self.store(path)
            update = first.learn(record(), observed(), DEFAULT, {'run': 'r1', 'trial': 'T0001'})
            second = self.store(path)
            cards = second.retrieve(observed(), DEFAULT)
            self.assertEqual(cards[0]['id'], update['card_id'])
            self.assertEqual(cards[0]['accepted'], 1)
            self.assertEqual(second.summary()['events'], 1)
            snapshot = json.loads((Path(path)/'offline/cards.json').read_text(encoding='utf-8'))
            self.assertEqual(snapshot['namespace'], 'offline')
            self.assertEqual(snapshot['cards'][0]['sources'][0]['run'], 'r1')
            self.assertNotIn('MODEL PROSE', json.dumps(snapshot))

    def test_negative_evidence_keeps_guard_failure_instead_of_becoming_success(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            store = self.store(path)
            store.learn(record(accepted=False), observed(), DEFAULT, {})
            card = store.retrieve(observed(), DEFAULT)[0]
            self.assertEqual(card['accepted'], 0)
            self.assertEqual(card['rejected'], 1)
            self.assertIn('parent:high_underestimation', card['last_guard_flags'])

    def test_deterministic_replays_do_not_inflate_evidence(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            store = self.store(path)
            self.assertEqual(store.learn(record(), observed(), DEFAULT, {'run': 'r1'})['status'], 'learned')
            self.assertEqual(store.learn(record(), observed(), DEFAULT, {'run': 'r2'})['status'], 'duplicate_ignored')
            self.assertEqual(store.summary()['events'], 1)

    def test_mixed_context_outcomes_remain_mixed_and_require_new_measurement(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            first = self.store(path)
            first.learn(record(), observed(), DEFAULT, {})
            context = dict(CONTEXT, context_id='context-two')
            second = self.store(path, context=context)
            second.learn(record(accepted=False), observed(), DEFAULT, {})
            card = second.retrieve(observed(), DEFAULT)[0]
            self.assertEqual((card['contexts'], card['accepted'], card['rejected']), (2, 1, 1))
            self.assertTrue(card['mixed_evidence'])
            self.assertIn('retest', card['use'])

    def test_domains_and_rule_versions_are_not_silently_transferred(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            first = self.store(path)
            first.learn(record(), observed(), DEFAULT, {})
            other = dict(CONTEXT, domain={'task': 'C', 'unit': 'minutes'})
            self.assertEqual(self.store(path, context=other).retrieve(observed(), DEFAULT), [])
            changed_rules = dict(asdict(Rules()), harmful_bias_allowance_cycles=3)
            second = ExperienceStore(path, 'offline', CONTEXT, changed_rules)
            self.assertEqual(second.retrieve(observed(), DEFAULT), [])

    def test_live_and_offline_databases_are_isolated_and_provenance_is_checked(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            offline, live = self.store(path), self.store(path, 'live')
            offline.learn(record(), observed(), DEFAULT, {})
            self.assertEqual(live.summary()['events'], 0)
            self.assertEqual(live.learn(record(), observed(), DEFAULT, {})['status'], 'skipped_provenance')
            self.assertEqual(offline.learn(record(simulated=False), observed(), DEFAULT, {})['status'], 'skipped_provenance')

    def test_cached_keep_invalid_failed_and_scope_mismatched_trials_never_become_cards(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            store = self.store(path)
            cases = []
            cached = record(); cached.update(reused=True, new_training=False); cases.append(cached)
            keep = record(); keep.update(new_training=False); keep['proposal']['action'] = 'keep'; cases.append(keep)
            invalid = record(); invalid['proposal']['action'] = 'invented'; cases.append(invalid)
            failed = record(); failed['candidate'] = {'status': 'timeout'}; cases.append(failed)
            mismatch = record(); mismatch['candidate']['evaluation_scope'] = 'wrong-window'; cases.append(mismatch)
            for value in cases:
                result = store.learn(value, dict(observed(), evaluation_scope='current-window'), DEFAULT, {})
                self.assertTrue(result['status'].startswith('skipped'))
            self.assertEqual(store.summary()['events'], 0)

    def test_disabled_memory_does_not_touch_storage(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            root = Path(path)/'absent'
            store = self.store(root, enabled=False)
            self.assertEqual(store.learn(record(), observed(), DEFAULT, {})['status'], 'disabled')
            self.assertEqual(store.retrieve(observed(), DEFAULT), [])
            self.assertFalse(root.exists())

    def test_nonfinite_metrics_cannot_poison_the_database(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            store = self.store(path)
            value = record(); value['candidate']['metrics']['nasa_score'] = float('nan')
            with self.assertRaises(ValueError): store.learn(value, observed(), DEFAULT, {})
            self.assertEqual(store.summary()['events'], 0)

    def test_prompt_budget_does_not_return_truncated_json(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            store = self.store(path, max_chars=500)
            store.learn(record(), observed(), DEFAULT, {})
            result = store.retrieve(observed(), DEFAULT)
            self.assertLessEqual(len(json.dumps(result, ensure_ascii=False, separators=(',', ':'))), 500)

    def test_concurrent_duplicate_writes_commit_one_event(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            self.store(path)
            def write(_):
                return self.store(path).learn(record(), observed(), DEFAULT, {})['status']
            with ThreadPoolExecutor(max_workers=2) as pool:
                statuses = list(pool.map(write, range(2)))
            self.assertEqual(sorted(statuses), ['duplicate_ignored', 'learned'])
            self.assertEqual(self.store(path).summary()['events'], 1)

    def test_semantic_context_ignores_container_timestamps_but_tracks_values(self):
        from types import SimpleNamespace
        model = SimpleNamespace(meta={'task': 'regression', 'paths': ['sensor'], 'hidden': 1, 'target_unit': 'cycles',
                                     'target_mean': 100., 'target_scale': 50., 'epochs': 15, 'steps': 10},
                                mean=np.zeros(1), scale=np.ones(1), params={'w': np.ones(1)},
                                first={'w': np.zeros(1)}, second={'w': np.zeros(1)})
        selection = {'x': np.array([[1.], [2.]]), 'y': np.array([3., 4.]), 'ids': ['a', 'b']}
        one = memory_context(model, 'train', selection, 'initial', 'actions')
        model.meta['container_timestamp'] = 'unimportant'
        self.assertEqual(one, memory_context(model, 'train', selection, 'initial', 'actions'))
        changed = copy.deepcopy(selection); changed['y'][0] += 1
        two = memory_context(model, 'train', changed, 'initial', 'actions')
        self.assertEqual(one['domain'], two['domain'])
        self.assertNotEqual(one['context_id'], two['context_id'])
        model.params['w'][0] += 1
        self.assertNotEqual(one['context_id'], memory_context(model, 'train', selection, 'initial', 'actions')['context_id'])


class ControllerMemoryTests(unittest.TestCase):
    def test_next_turn_and_new_run_retrieve_cards_and_log_citation(self):
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            root = Path(path)
            memory = ExperienceStore(root/'library', 'offline', CONTEXT, asdict(Rules()))
            first = Controller(root/'run-one', FakeBackend(), experience=memory)
            first.step(FixedPlanner('pick_best_epoch'), 1)
            cards = first.observe()['experience']['cards']
            self.assertEqual(len(cards), 1)
            card_id = cards[0]['id']
            second = Controller(root/'run-two', FakeBackend(), experience=memory)
            self.assertIn(card_id, second.observe()['available_evidence_ids'])
            class CitingPlanner(FixedPlanner):
                def propose(self, observation, turn):
                    return json.dumps({'action': 'lr_half', 'why': 'use measured evidence', 'used': [card_id]}), {'simulated': True}
            row = second.step(CitingPlanner('lr_half'), 1)
            self.assertEqual(row['experience_used'], [card_id])
            self.assertTrue((root/'run-two/experience-updates/001.json').exists())

    def test_broken_optional_memory_does_not_change_the_training_decision(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=TEMP) as path:
            def fail(*args): raise sqlite3.OperationalError('fixture failure')
            memory = SimpleNamespace(retrieve=fail, enabled=True, namespace='offline', inserted=0, duplicates=0, path=Path(path)/'db')
            controller = Controller(path, FakeBackend(), experience=memory)
            row = controller.step(FixedPlanner('pick_best_epoch'), 1)
            self.assertTrue(row['decision']['accepted'])
            self.assertTrue(controller.snapshot()['experience']['errors'])


class ImportProofTests(unittest.TestCase):
    def proof(self):
        proposal = {'action': 'pick_best_epoch', 'why': 'measured curve', 'used': ['BASE-REFERENCE']}
        call = {'mode': 'live', 'status': 'ok', 'transport_simulated': False, 'finish_reason': 'stop',
                'returned_model': 'Qwen/Qwen3.8-27B', 'provider_id': 'chatcmpl-unit-fixture',
                'usage': {'total_tokens': 20}, 'raw': json.dumps(proposal)}
        trial = {'proposal': proposal, 'simulated_planner': False, 'provider': {'simulated': False}}
        return call, {'status': 'ok'}, trial, {'available_evidence_ids': ['BASE-REFERENCE']}

    def test_legacy_settings_are_completed_but_arbitrary_fields_rejected(self):
        old = asdict(DEFAULT); old.pop('epochs'); old.pop('loss_mode')
        self.assertEqual(complete_settings(old), asdict(DEFAULT))
        with self.assertRaises(ValueError): complete_settings(dict(old, unknown=True))

    def test_provider_proof_requires_consistent_response_action(self):
        values = self.proof()
        valid_live_proof(*values)
        values[2]['proposal'] = {'action': 'keep', 'why': 'altered', 'used': []}
        with self.assertRaises(ValueError): valid_live_proof(*values)

    def test_simulated_truncated_missing_usage_or_wrong_model_cannot_be_imported(self):
        for key, value in [('transport_simulated', True), ('finish_reason', 'length'), ('usage', {}),
                           ('returned_model', 'another-model'), ('provider_id', 'FIXTURE-NO-NETWORK')]:
            values = self.proof(); values[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError): valid_live_proof(*values)


if __name__ == '__main__':
    unittest.main()
