"""Training policy for continuing a CPU NumPy MLP checkpoint.

Only the code between the two EVOLVE-BLOCK marker comments may be changed.
The frozen runner (harness/runner.py) calls CONFIG and the five hooks below; their
names, arguments and return shapes must stay as they are (contract: docs/HOOKS.md,
also given in the system prompt). Only numpy (np) and math may be used.
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
    return base_lr


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
    return len(history) - 1
# EVOLVE-BLOCK-END
