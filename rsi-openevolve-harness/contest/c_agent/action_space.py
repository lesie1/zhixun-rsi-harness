"""Deterministic, composable training actions; model replies never become Python code."""
import ast
from dataclasses import asdict, dataclass, replace
import hashlib
import json


@dataclass(frozen=True)
class Settings:
    lr_scale: float = 1.0
    lr_schedule: str = 'constant'
    batch_size: int | None = None
    weight_mode: str = 'uniform'
    best_epoch: bool = False
    epochs: int = 5
    loss_mode: str = 'mse'


DEFAULT = Settings()
MENU = {
    'keep': 'No new training. Preserve the real parent checkpoint as a fallback; do not erase an already measured search winner.',
    'continue_default': 'Reset policy settings to official five-epoch continuation. Weights still resume from the fixed stage parent.',
    'pick_best_epoch': 'Select the epoch with minimum selection RMSE; still execute all requested continuation epochs.',
    'lr_half': 'Set learning rate multiplier to 0.5; preserve other settings. Repeating does not halve it again.',
    'batch_512': 'Set batch size to 512; preserve other settings.',
    'high_rul_weight_1_2': 'Give training targets at/above training Q75 weight 1.2, other rows 1, then normalize mean to one.',
    'low_rul_weight_1_2': 'Give training targets at/below training Q25 weight 1.2, other rows 1, then normalize mean to one.',
    'both_tail_weight_1_2': 'Give both training target tails weight 1.2, middle rows 1, then normalize mean to one.',
    'lr_decay': 'Use a linear learning-rate decay from current multiplier to half that rate over the requested epochs.',
    'epochs_1': 'Conservative probe: reset to uniform-weight MSE, one epoch, quarter base learning rate, and best-epoch selection. Not a forced adoption.',
    'epochs_3': 'Conservative probe: reset to uniform-weight MSE, three epochs, quarter base learning rate, and best-epoch selection.',
    'tail_error_1_1': 'Enable best-epoch selection and multiply squared-error loss/gradient by 1.1 only for low-target overprediction or high-target underprediction, using training-batch target quartiles. Keep other settings/errors unchanged.',
}


def validate_settings(value):
    if isinstance(value, Settings):
        value = asdict(value)
    if not isinstance(value, dict) or set(value) != set(asdict(DEFAULT)):
        raise ValueError('Unexpected settings fields')
    if type(value['lr_scale']) not in (int, float) or value['lr_scale'] not in (0.25, 0.5, 1.0):
        raise ValueError('Invalid learning-rate multiplier')
    if value['lr_schedule'] not in ('constant', 'linear_half'):
        raise ValueError('Invalid schedule')
    if value['batch_size'] is not None and (type(value['batch_size']) is not int or value['batch_size'] != 512):
        raise ValueError('Invalid batch size')
    if value['weight_mode'] not in ('uniform', 'high', 'low', 'both'):
        raise ValueError('Invalid weight mode')
    if type(value['best_epoch']) is not bool:
        raise ValueError('Invalid epoch-selection flag')
    if type(value['epochs']) is not int or value['epochs'] not in (1, 3, 5):
        raise ValueError('Invalid continuation epoch count')
    if value['loss_mode'] not in ('mse', 'tail_error_1_1'):
        raise ValueError('Invalid loss mode')
    return Settings(**value)


def apply_action(action, current):
    current = validate_settings(current)
    if not isinstance(action, str) or action not in MENU:
        raise ValueError('Unknown action')
    if action == 'keep':
        return None
    if action in ('epochs_1', 'epochs_3'):
        return replace(DEFAULT, epochs=1 if action == 'epochs_1' else 3, lr_scale=0.25, best_epoch=True)
    changes = {
        'pick_best_epoch': {'best_epoch': True},
        'lr_half': {'lr_scale': 0.5},
        'lr_decay': {'lr_schedule': 'linear_half'},
        'batch_512': {'batch_size': 512},
        'high_rul_weight_1_2': {'weight_mode': 'high'},
        'low_rul_weight_1_2': {'weight_mode': 'low'},
        'both_tail_weight_1_2': {'weight_mode': 'both'},
        'tail_error_1_1': {'loss_mode': 'tail_error_1_1', 'best_epoch': True},
    }
    return DEFAULT if action == 'continue_default' else replace(current, **changes[action])


def settings_id(settings):
    value = asdict(validate_settings(settings))
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def render_policy(initial, settings):
    settings = validate_settings(settings)
    if settings == DEFAULT:
        return initial
    config = (f'CONFIG = {{\n    "epochs": {settings.epochs},\n'
              f'    "batch_size": {settings.batch_size!r},\n    "grad_clip": 5.0,\n}}')
    lr = ('def learning_rate(epoch, total_epochs, base_lr):\n'
          f'    rate = float(base_lr * {settings.lr_scale!r})\n')
    if settings.lr_schedule == 'linear_half':
        lr += '    if total_epochs > 1:\n        rate *= 1.0 - 0.5 * epoch / (total_epochs - 1)\n'
    lr += '    return rate'
    weights = ('def sample_weights(y, task, num_classes):\n'
               '    if task != "regression":\n        return np.ones(len(y))\n')
    if settings.weight_mode == 'uniform':
        weights += '    return np.ones(len(y))'
    else:
        weights += '    q25, q75 = np.quantile(y, [0.25, 0.75])\n'
        mask = {'low': 'y <= q25', 'high': 'y >= q75', 'both': '(y <= q25) | (y >= q75)'}[settings.weight_mode]
        weights += f'    w = np.where({mask}, 1.2, 1.0)\n    return w / w.mean()'
    pick = ('def pick_epoch(history):\n    return int(np.argmin([h["val"]["rmse"] for h in history]))'
            if settings.best_epoch else 'def pick_epoch(history):\n    return len(history) - 1')
    replacements = {'CONFIG': config, 'learning_rate': lr, 'sample_weights': weights, 'pick_epoch': pick}
    tree = ast.parse(initial)
    if settings.loss_mode == 'tail_error_1_1':
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'output_grad')
        original_grad = ast.get_source_segment(initial, node)
        old = '    loss = np.sum(w * err ** 2)\n    return loss, (2 * w * err / len(target))[:, None]'
        if original_grad.count(old) != 1:
            raise ValueError('Unrecognized official regression loss; refusing to patch it')
        new = ('    q25, q75 = np.quantile(target, [0.25, 0.75])\n'
               '    harmful = (q25 < q75) & (((target <= q25) & (err > 0)) | ((target >= q75) & (err < 0)))\n'
               '    effective = w * np.where(harmful, 1.1, 1.0)\n'
               '    loss = np.sum(effective * err ** 2)\n'
               '    return loss, (2 * effective * err / len(target))[:, None]')
        replacements['output_grad'] = original_grad.replace(old, new)
    edits = []
    for node in tree.body:
        name = node.name if isinstance(node, ast.FunctionDef) else None
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in node.targets):
            name = 'CONFIG'
        if name in replacements:
            edits.append((node.lineno - 1, node.end_lineno, replacements.pop(name)))
    if replacements:
        raise ValueError('Initial policy is missing required hooks')
    lines = initial.splitlines()
    for start, end, code in sorted(edits, reverse=True):
        lines[start:end] = code.splitlines()
    result = '\n'.join(lines) + ('\n' if initial.endswith('\n') else '')
    ast.parse(result)
    if initial.split('# EVOLVE-BLOCK-START')[0] != result.split('# EVOLVE-BLOCK-START')[0]:
        raise ValueError('Modification escaped the permitted block')
    if initial.split('# EVOLVE-BLOCK-END')[1] != result.split('# EVOLVE-BLOCK-END')[1]:
        raise ValueError('Modification escaped the permitted block')
    return result
