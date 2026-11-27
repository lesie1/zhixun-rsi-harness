"""Training policy for continuing a CPU NumPy MLP checkpoint (contest image, candidate "pick").

Origin: the C seed policy found by the team's OpenEvolve search (Y/C_training/20261124-v1/policy_seed.py,
sha256 0f7e38a4…): learning rate decays linearly to half, and the epoch with the best public-validation
score is kept. Generalised here to classification (best macro-F1) so one file serves every task.
Validation is only used to choose the epoch, never for gradients.
"""
import math

import numpy as np

# EVOLVE-BLOCK-START
CONFIG = {
    "epochs": 5,
    "batch_size": None,
    "grad_clip": 5.0,
}


def learning_rate(epoch, total_epochs, base_lr):
    if total_epochs <= 1:
        return base_lr
    frac = 1.0 - 0.5 * (epoch / (total_epochs - 1))
    return base_lr * frac


def sample_weights(y, task, num_classes):
    return np.ones(len(y))


def output_grad(out, target, w, task):
    if task == "classification":
        z = out - out.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        rows = np.arange(len(target))
        loss = -np.sum(w * np.log(np.maximum(p[rows, target], 1e-300)))
        grad = p.copy()
        grad[rows, target] -= 1
        return loss, grad * w[:, None] / len(target)
    err = out[:, 0] - target
    loss = np.sum(w * err ** 2)
    return loss, (2 * w * err / len(target))[:, None]


def extra_grad(params):
    return {name: np.zeros_like(value) for name, value in params.items()}


def pick_epoch(history):
    best_idx, best = 0, None
    for i, h in enumerate(history):
        val = h["val"]
        score = -val["rmse"] if "rmse" in val else val.get("macro_f1", 0.0)
        if best is None or score > best:
            best, best_idx = score, i
    return best_idx
# EVOLVE-BLOCK-END
