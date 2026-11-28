import ast
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from action_space import DEFAULT, MENU, Settings, apply_action, render_policy, validate_settings
from backend import Candidate, evaluation_scope
from controller import Controller
from feedback import Rules, extra_violations, parse_proposal, select
from planner import BudgetExhausted, LivePlanner
from c_actions import freeze, gate, gate_choice

INITIAL = (ROOT / 'initial_policy.py').read_text(encoding='utf-8')
TEST_TEMP = ROOT / '.test-tmp'
TEST_TEMP.mkdir(exist_ok=True)


def observed(rmse=50, mae=35, nasa=100, low_bias=18, high_bias=-44, low_rmse=32, high_rmse=65):
    return {'status': 'ok', 'metrics': {'rmse': rmse, 'mae': mae, 'nasa_score': nasa},
            'target_slices': [{'low': 0, 'high': 50, 'n': 25, 'rmse': low_rmse, 'bias': low_bias},
                              {'low': 150, 'high': 400, 'n': 25, 'rmse': high_rmse, 'bias': high_bias}],
            'source_subsets': [{'group': 'FD001', 'rmse': 40}, {'group': 'FD004', 'rmse': 60}],
            'core_guard_ok': True, 'core_violations': []}


class ActionTests(unittest.TestCase):
    def namespace(self, settings):
        namespace = {}
        exec(compile(render_policy(INITIAL, settings), '<generated checked policy>', 'exec'), namespace)
        return namespace

    def test_default_is_byte_identical_to_official_policy(self):
        self.assertEqual(render_policy(INITIAL, DEFAULT), INITIAL)

    def test_actions_preserve_regularizer_and_code_outside_block(self):
        original = ast.parse(INITIAL)
        original_functions = {n.name: ast.dump(n) for n in original.body if isinstance(n, ast.FunctionDef)}
        for action in MENU:
            settings = apply_action(action, DEFAULT)
            if settings is None:
                continue
            source = render_policy(INITIAL, settings)
            funcs = {n.name: ast.dump(n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
            for name in ['output_grad', 'extra_grad']:
                if name == 'output_grad' and settings.loss_mode != 'mse':
                    continue
                self.assertEqual(funcs[name], original_functions[name], action)
            self.assertEqual(source.split('# EVOLVE-BLOCK-START')[0], INITIAL.split('# EVOLVE-BLOCK-START')[0])
            self.assertEqual(source.split('# EVOLVE-BLOCK-END')[1], INITIAL.split('# EVOLVE-BLOCK-END')[1])

    def test_composition_is_idempotent_and_default_really_resets(self):
        state = apply_action('pick_best_epoch', DEFAULT)
        state = apply_action('lr_half', state)
        self.assertEqual(apply_action('lr_half', state), state)
        self.assertEqual(state.lr_scale, 0.5)
        self.assertTrue(state.best_epoch)
        self.assertEqual(apply_action('continue_default', state), DEFAULT)

    def test_weight_actions_replace_instead_of_stack(self):
        high = apply_action('high_rul_weight_1_2', DEFAULT)
        both = apply_action('both_tail_weight_1_2', high)
        self.assertEqual(both, apply_action('both_tail_weight_1_2', DEFAULT))

    def test_raw_training_target_quantiles_and_mean_one_weights(self):
        y = np.array([0, 10, 20, 30, 100, 200, 400, 800.])
        for action in ['high_rul_weight_1_2', 'low_rul_weight_1_2', 'both_tail_weight_1_2']:
            module = self.namespace(apply_action(action, DEFAULT))
            w = module['sample_weights'](y, 'regression', 0)
            self.assertAlmostEqual(w.mean(), 1)
            self.assertTrue(np.all(np.isfinite(w)))
            if action.startswith('high'):
                self.assertGreater(w[-1], w[3])
                self.assertEqual(w[0], w[3])
            elif action.startswith('low'):
                self.assertGreater(w[0], w[3])
                self.assertEqual(w[-1], w[3])
            else:
                self.assertGreater(w[0], w[3])
                self.assertGreater(w[-1], w[3])

    def test_constant_target_ties_do_not_explode_weights(self):
        for action in ['high_rul_weight_1_2', 'low_rul_weight_1_2', 'both_tail_weight_1_2']:
            w = self.namespace(apply_action(action, DEFAULT))['sample_weights'](np.ones(8)*200, 'regression', 0)
            np.testing.assert_allclose(w, np.ones(8))

    def test_learning_rates_and_selected_epoch(self):
        state = apply_action('lr_decay', apply_action('lr_half', apply_action('pick_best_epoch', DEFAULT)))
        module = self.namespace(state)
        self.assertAlmostEqual(module['learning_rate'](0, 5, .002), .001)
        self.assertAlmostEqual(module['learning_rate'](4, 5, .002), .0005)
        self.assertAlmostEqual(module['learning_rate'](0, 1, .002), .001)
        history = [{'val': {'rmse': x}} for x in [50, 49, 48, 48, 49]]
        self.assertEqual(module['pick_epoch'](history), 2)

    def test_untrusted_settings_cannot_escape_menu(self):
        for state in [dict(asdict(DEFAULT), lr_scale=True), dict(asdict(DEFAULT), batch_size=True),
                      dict(asdict(DEFAULT), lr_schedule='__import__("os")'),
                      dict(asdict(DEFAULT), best_epoch=1), dict(asdict(DEFAULT), arbitrary='x'),
                      dict(asdict(DEFAULT), epochs=True), dict(asdict(DEFAULT), epochs=100),
                      dict(asdict(DEFAULT), loss_mode='arbitrary_code')]:
            with self.assertRaises(ValueError):
                validate_settings(state)
        with self.assertRaises(ValueError):
            apply_action('change_architecture', DEFAULT)

    def test_short_probes_and_reset_preserve_bounded_epoch_counts(self):
        for action, epochs in [('epochs_1', 1), ('epochs_3', 3)]:
            settings = apply_action(action, apply_action('pick_best_epoch', DEFAULT))
            module = self.namespace(settings)
            self.assertEqual(module['CONFIG']['epochs'], epochs)
            self.assertEqual(settings.lr_scale, 0.25)
            self.assertEqual(settings.weight_mode, 'uniform')
            self.assertEqual(settings.loss_mode, 'mse')
            self.assertEqual(module['pick_epoch']([{'val': {'rmse': 50-i}} for i in range(epochs)]), epochs-1)
            self.assertEqual(apply_action('continue_default', settings), DEFAULT)

    def test_tail_loss_only_penalizes_the_two_harmful_directions(self):
        module = self.namespace(apply_action('tail_error_1_1', DEFAULT))
        y = np.array([-4., -3., -2., -1., 1., 2., 3., 4.])
        err = np.array([.4, -.4, .4, -.4, .4, -.4, .4, -.4])
        loss, gradient = module['output_grad']((y+err)[:, None], y, np.ones(8), 'regression')
        expected = np.array([1.1, 1., 1., 1., 1., 1., 1., 1.1])
        np.testing.assert_allclose(gradient[:, 0], 2*expected*err/len(y))
        self.assertAlmostEqual(loss, np.sum(expected*err**2))

    def test_tail_gradient_matches_finite_difference_of_mean_loss(self):
        module = self.namespace(apply_action('tail_error_1_1', DEFAULT))
        y = np.array([-4., -3., -2., -1., 1., 2., 3., 4.])
        out = (y+np.array([.4, -.4, .6, -.6, .3, -.3, .5, -.5]))[:, None]
        w = np.array([1., 1.2, .8, 1., 1.1, .9, 1., 1.])
        _, gradient = module['output_grad'](out, y, w, 'regression')
        epsilon = 1e-6
        for index in range(len(y)):
            upper, lower = out.copy(), out.copy()
            upper[index, 0] += epsilon
            lower[index, 0] -= epsilon
            plus = module['output_grad'](upper, y, w, 'regression')[0]/len(y)
            minus = module['output_grad'](lower, y, w, 'regression')[0]/len(y)
            self.assertAlmostEqual(gradient[index, 0], (plus-minus)/(2*epsilon), places=8)

    def test_tail_partition_is_invariant_to_positive_target_scaling_and_shift(self):
        module = self.namespace(apply_action('tail_error_1_1', DEFAULT))
        y = np.array([-4., -3., -2., -1., 1., 2., 3., 4.])
        out = (y+.4)[:, None]
        loss, gradient = module['output_grad'](out, y, np.ones(8), 'regression')
        transformed_loss, transformed_gradient = module['output_grad'](out*7+100, y*7+100, np.ones(8), 'regression')
        self.assertAlmostEqual(transformed_loss, loss*49)
        np.testing.assert_allclose(transformed_gradient, gradient*7)

    def test_tail_loss_constant_targets_do_not_invent_lifetime_tails(self):
        module = self.namespace(apply_action('tail_error_1_1', DEFAULT))
        y = np.ones(8)
        loss, gradient = module['output_grad']((y+.5)[:, None], y, np.ones(8), 'regression')
        self.assertAlmostEqual(loss, 2.)
        np.testing.assert_allclose(gradient, np.ones((8, 1))/8)

    def test_tail_action_does_not_change_classification_gradient(self):
        official = self.namespace(DEFAULT)
        tail = self.namespace(apply_action('tail_error_1_1', DEFAULT))
        out = np.array([[1., 0.], [.3, .6], [.9, .2]])
        y = np.array([0, 1, 1])
        a = official['output_grad'](out, y, np.ones(3), 'classification')
        b = tail['output_grad'](out, y, np.ones(3), 'classification')
        self.assertEqual(a[0], b[0])
        np.testing.assert_allclose(a[1], b[1])


class ProposalTests(unittest.TestCase):
    def test_valid_proposal_and_evidence(self):
        obj = parse_proposal('{"action":"lr_half","why":"measured instability","used":["T0001"]}', ['T0001'])
        self.assertEqual(obj['action'], 'lr_half')

    def test_code_extra_fields_duplicates_nonfinite_unknown_action_rejected(self):
        cases = ['print("hello")', '{"action":"keep","why":"x","used":[],"code":"import os"}',
                 '{"action":"keep","action":"lr_half","why":"x","used":[]}',
                 '{"action":"keep","why":NaN,"used":[]}',
                 '{"action":"change_architecture","why":"x","used":[]}',
                 '{"action":"keep","why":"x","used":["invented"]}',
                 '{"action":"keep","why":"x","used":["T0001","T0001"]}']
        for value in cases:
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                parse_proposal(value, ['T0001'])


class SelectionTests(unittest.TestCase):
    def test_scope_changes_when_labels_change_even_with_identical_sample_ids(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            train, selection = Path(temp)/'train', Path(temp)/'selection'
            train.write_bytes(b'train window')
            selection.write_bytes(b'features and labels in window one')
            one = evaluation_scope('parent-one', train, selection, ['same-id'])
            selection.write_bytes(b'features and labels in window two')
            two = evaluation_scope('parent-one', train, selection, ['same-id'])
            self.assertNotEqual(one, two)
            self.assertNotEqual(two, evaluation_scope('parent-two', train, selection, ['same-id']))

    def test_cross_window_or_parent_comparison_is_rejected_before_selection(self):
        parent = dict(observed(), evaluation_scope='current-window-parent')
        incumbent = dict(observed(rmse=49), evaluation_scope='current-window-parent')
        candidate = dict(observed(rmse=45), evaluation_scope='old-window-parent')
        self.assertEqual(select(candidate, incumbent, parent, Rules())['reason'], 'evaluation_scope_mismatch')
        candidate['evaluation_scope'] = 'current-window-parent'
        self.assertTrue(select(candidate, incumbent, parent, Rules())['accepted'])
        del candidate['evaluation_scope']
        self.assertFalse(select(candidate, incumbent, parent, Rules())['accepted'])

    def test_valid_primary_improvement_accepted(self):
        self.assertTrue(select(observed(rmse=49), observed(), observed(), Rules())['accepted'])

    def test_low_rul_overestimation_cannot_be_bought_with_overall_gain(self):
        result = select(observed(rmse=45, low_bias=23), observed(), observed(), Rules())
        self.assertFalse(result['accepted'])
        self.assertTrue(any('low_overestimation' in x for x in result['violations']))

    def test_high_rul_underestimation_guard(self):
        result = select(observed(rmse=45, high_bias=-48), observed(), observed(), Rules())
        self.assertFalse(result['accepted'])
        self.assertTrue(any('high_underestimation' in x for x in result['violations']))

    def test_tail_error_group_and_missing_group_protected(self):
        for candidate in [observed(rmse=45, high_rmse=70), observed(rmse=45, low_rmse=35)]:
            self.assertFalse(select(candidate, observed(), observed(), Rules())['accepted'])
        candidate = observed(rmse=45)
        candidate['source_subsets'] = candidate['source_subsets'][:1]
        self.assertFalse(select(candidate, observed(), observed(), Rules())['accepted'])

    def test_auxiliary_gain_cannot_compensate_rmse_loss_or_tiny_gain(self):
        self.assertFalse(select(observed(rmse=50.01, nasa=1), observed(), observed(), Rules())['accepted'])
        self.assertFalse(select(observed(rmse=49.99), observed(), observed(), Rules())['accepted'])

    def test_true_parent_and_incumbent_both_protected(self):
        parent = observed(low_bias=10)
        incumbent = observed(rmse=49, low_bias=11)
        result = select(observed(rmse=48, low_bias=13.5), incumbent, parent, Rules())
        self.assertFalse(result['accepted'])


class FakeBackend:
    def __init__(self):
        self.calls = 1
        self.parent = Candidate('parent', None, observed(), '/parent/checkpoint.npz', None)
        self.reference = Candidate('reference', DEFAULT, observed(rmse=49), '/reference/checkpoint.npz', 'initial.py')
        self.cache = {DEFAULT: self.reference}

    def evaluate(self, settings):
        if settings in self.cache:
            return self.cache[settings], True
        self.calls += 1
        self.cache[settings] = Candidate(str(self.calls), settings, observed(rmse=48), '/candidate/checkpoint.npz', 'policy.py')
        return self.cache[settings], False


class GateTests(unittest.TestCase):
    def core(self, decision='adopt_candidate', reference_passes=True):
        return {'decision': decision, 'reason': 'core evidence',
                'comparisons': {'reference_vs_parent': {'passes': reference_passes}}}

    def test_candidate_tail_failure_falls_back_to_safe_reference_with_matching_reason(self):
        compact = {'parent': observed(), 'reference': observed(rmse=49),
                   'candidate': observed(rmse=45, low_bias=23)}
        decision, reason, extra = gate_choice(self.core(), compact, Rules())
        self.assertEqual(decision, 'adopt_reference')
        self.assertIn('官方续训', reason)
        self.assertIn('low_overestimation', extra['candidate'])

    def test_unsafe_reference_never_becomes_rollback_choice(self):
        compact = {'parent': observed(), 'reference': observed(rmse=49, high_bias=-48),
                   'candidate': observed(rmse=45, low_bias=23)}
        self.assertEqual(gate_choice(self.core(), compact, Rules())[0], 'keep_parent')
        self.assertEqual(gate_choice(self.core('adopt_reference'), compact, Rules())[0], 'keep_parent')

    def test_clean_candidate_preserves_original_gate_decision_and_reason(self):
        compact = {'parent': observed(), 'reference': observed(rmse=49),
                   'candidate': observed(rmse=48)}
        decision, reason, extra = gate_choice(self.core(), compact, Rules())
        self.assertEqual((decision, reason), ('adopt_candidate', 'core evidence'))
        self.assertTrue(all(not value for value in extra.values()))

    def test_freeze_rejects_offline_and_detects_tampering_before_opening_gate(self):
        from backend import write_json
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            run = Path(temp)
            write_json(run/'manifest.json', {'mode': 'offline', 'status': 'completed'})
            with self.assertRaises(ValueError): freeze(run)
            checkpoint = run/'checkpoint.npz'
            checkpoint.write_bytes(b'fake checkpoint for integrity test')
            policy = run/'policy.py'
            policy.write_text('policy fixture')
            write_json(run/'manifest.json', {'mode': 'live', 'status': 'completed'})
            write_json(run/'state.json', {'incumbent': {'checkpoint': str(checkpoint), 'policy': str(policy)},
                                         'rules': asdict(Rules())})
            write_json(run/'config.json', {'fixture': True})
            write_json(run/'protected-after.json', {})
            freeze(run)
            with self.assertRaises(ValueError): freeze(run)
            checkpoint.write_bytes(b'changed after freezing')
            with self.assertRaisesRegex(ValueError, 'Frozen experiment changed'): gate(run)
            self.assertFalse((run/'gate-started.json').exists())


class FixedPlanner:
    simulated = True
    def __init__(self, action):
        self.action = action
        self.observations = []
    def propose(self, observation, turn):
        self.observations.append(observation)
        return json.dumps({'action': self.action, 'why': 'fixture', 'used': []}), {'simulated': True}


class LoopTests(unittest.TestCase):
    def test_unmeasured_menu_and_keep_streak_are_visible_without_forcing_training(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            backend = FakeBackend()
            controller = Controller(temp, backend)
            self.assertIn('epochs_1', controller.observe()['unmeasured_actions_from_current_settings'])
            controller.step(FixedPlanner('pick_best_epoch'), 1)
            self.assertIn('pick_best_epoch', controller.observe()['already_measured_actions_from_current_settings'])
            count = backend.calls
            controller.step(FixedPlanner('keep'), 2)
            controller.step(FixedPlanner('keep'), 3)
            self.assertEqual(controller.observe()['keep_streak'], 2)
            self.assertEqual(backend.calls, count)

    def test_keep_does_not_train_or_erase_measured_winner(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            backend = FakeBackend()
            controller = Controller(temp, backend)
            winner = controller.incumbent
            row = controller.step(FixedPlanner('keep'), 1)
            self.assertEqual(backend.calls, 1)
            self.assertEqual(controller.incumbent, winner)
            self.assertFalse(row['new_training'])
            self.assertEqual(row['decision']['parent_checkpoint'], '/parent/checkpoint.npz')

    def test_second_turn_reads_feedback_and_duplicate_is_cached(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            backend = FakeBackend()
            controller = Controller(temp, backend)
            planner = FixedPlanner('pick_best_epoch')
            first = controller.step(planner, 1)
            count = backend.calls
            second = controller.step(planner, 2)
            self.assertTrue(first['decision']['accepted'])
            self.assertEqual(planner.observations[1]['previous_trials'][0]['id'], 'T0001')
            self.assertIn('T0001', planner.observations[1]['available_evidence_ids'])
            self.assertEqual(backend.calls, count)
            self.assertTrue(second['reused'])

    def test_invalid_reply_is_retained_and_never_trains(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            backend = FakeBackend()
            controller = Controller(temp, backend)
            controller.step(FixedPlanner('change_architecture'), 1)
            self.assertEqual(backend.calls, 1)
            self.assertIn('invalid_reply', controller.observe()['previous_trials'][0]['decision']['reason'])
            self.assertEqual(controller.observe()['gate_metrics'], 'unavailable')


class FakeClient:
    def __init__(self, finish='stop'):
        self.kwargs = []
        self.finish = finish
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        return SimpleNamespace(usage=SimpleNamespace(model_dump=lambda: {'total_tokens': 20}),
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"action":"keep","why":"test","used":[]}'),
                                     finish_reason=self.finish)], model='Qwen/Qwen3.8-27B', id='FIXTURE-NO-NETWORK')
    def __enter__(self): return self
    def __exit__(self, *args): return False


class TransportTests(unittest.TestCase):
    def test_one_request_budget_json_mode_and_thinking_disabled(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            client = FakeClient()
            planner = LivePlanner(temp, 1, max_total_tokens=100000, client_factory=lambda: client)
            raw, metadata = planner.propose({'task': 'test'}, 1)
            self.assertTrue(metadata['simulated'])
            self.assertEqual(client.kwargs[0]['extra_body'], {'enable_thinking': False})
            self.assertEqual(client.kwargs[0]['response_format'], {'type': 'json_object'})
            with self.assertRaises(BudgetExhausted): planner.propose({}, 2)
            self.assertEqual(len(client.kwargs), 1)
            with self.assertRaises(ValueError): LivePlanner(temp, 1, client_factory=lambda: client)

    def test_truncation_has_no_retry_and_keeps_usage(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            client = FakeClient(finish='length')
            planner = LivePlanner(temp, 1, max_total_tokens=100000, client_factory=lambda: client)
            with self.assertRaises(RuntimeError): planner.propose({}, 1)
            self.assertEqual(len(client.kwargs), 1)
            ledger = json.loads((Path(temp)/'provider-ledger.json').read_text())
            self.assertEqual(ledger['1']['charged'], 20)
            self.assertEqual(ledger['1']['status'], 'ValueError')

    def test_insufficient_token_budget_sends_no_request(self):
        with tempfile.TemporaryDirectory(dir=TEST_TEMP) as temp:
            client = FakeClient()
            planner = LivePlanner(temp, 1, max_total_tokens=1, client_factory=lambda: client)
            with self.assertRaises(BudgetExhausted): planner.propose({}, 1)
            self.assertEqual(client.kwargs, [])


if __name__ == '__main__':
    unittest.main()
