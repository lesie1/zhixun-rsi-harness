#!/usr/bin/env python
"""A fake OpenAI-compatible chat endpoint for testing the whole pipeline without a GPU.

It is NOT a language model. It reads the current policy out of the prompt and answers
with one scripted SEARCH/REPLACE edit, cycling through sensible edits and deliberate
mistakes (a SEARCH that cannot match, an edit outside the evolve block, a forbidden
import, a hook returning an invalid value), so every path of the harness gets exercised.

  python scripts/mock_llm_server.py --port 8765 --log-dir /tmp/mock_prompts
  python scripts/run_evolve.py --run-config configs/run_A.json --iterations 12 \
      --api-base http://127.0.0.1:8765/v1 --model mock
"""
import argparse
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

_lock = threading.Lock()
_state = {"calls": 0, "log_dir": None}


def _block(search, replace):
    return f"<<<<<<< SEARCH\n{search}\n=======\n{replace}\n>>>>>>> REPLACE"


def _edit_weights(code):
    old = "def sample_weights(y, task, num_classes):\n    return np.ones(len(y))"
    new = ("def sample_weights(y, task, num_classes):\n"
           "    if task != \"classification\":\n"
           "        return np.ones(len(y))\n"
           "    counts = np.bincount(y, minlength=num_classes).astype(float)\n"
           "    w = (len(y) / (num_classes * np.maximum(counts, 1.0)))[y]\n"
           "    return w / w.mean()")
    if old in code:
        return ("少数类样本太少导致召回率低，按类别频率的倒数给 sample_weights 加权，预期少数类召回率上升",
                _block(old, new))


def _edit_pick(code):
    old = "def pick_epoch(history):\n    return len(history) - 1"
    new = ("def pick_epoch(history):\n"
           "    if \"macro_f1\" in history[0][\"val\"]:\n"
           "        return int(np.argmax([h[\"val\"][\"macro_f1\"] for h in history]))\n"
           "    return int(np.argmin([h[\"val\"][\"rmse\"] for h in history]))")
    if old in code:
        return ("曲线显示最后一轮不是最好的一轮，用 pick_epoch 选挑选集上最好的一轮", _block(old, new))


def _edit_epochs(code):
    match = re.search(r'    "epochs": (\d+),', code)
    if match and int(match.group(1)) < 12:
        return ("指标到最后一轮仍在变好，增加训练轮数",
                _block(match.group(0), f'    "epochs": {int(match.group(1)) + 3},'))


def _edit_lr(code):
    old = "def learning_rate(epoch, total_epochs, base_lr):\n    return base_lr"
    new = ("def learning_rate(epoch, total_epochs, base_lr):\n"
           "    progress = epoch / max(1, total_epochs - 1)\n"
           "    return base_lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress)))")
    if old in code:
        return ("后几轮指标来回波动，让学习率按余弦衰减，预期曲线更平稳", _block(old, new))


def _edit_l2(code):
    old = "    return {name: np.zeros_like(value) for name, value in params.items()}"
    new = ("    return {name: (1e-4 * value if name in (\"w1\", \"w2\") else np.zeros_like(value))\n"
           "            for name, value in params.items()}")
    if old in code:
        return ("训练损失下降而挑选集指标变差，用 extra_grad 给权重加 L2 正则", _block(old, new))


def _bad_search(code):
    return ("（故意出错）SEARCH 与原文不一致",
            _block("def sample_weights(y, task):\n    return np.ones(y.shape)", "def sample_weights(y, task):\n    return y"))


def _bad_outside(code):
    inside = _edit_lr(code) or _edit_epochs(code)
    outside = _block("import math\n\nimport numpy as np", "import math\nimport os\n\nimport numpy as np")
    return ("（故意出错）同时改了 EVOLVE-BLOCK 外面的 import", outside + ("\n\n" + inside[1] if inside else ""))


def _bad_import(code):
    old = "def pick_epoch(history):"
    if old in code:
        return ("（故意出错）在块内 import 不允许的模块", _block(old, "import os\n\n\ndef pick_epoch(history):"))


def _bad_value(code):
    match = re.search(r'    "grad_clip": ([0-9.]+),', code)
    if match:
        return ("（故意出错）把梯度裁剪设成负数", _block(match.group(0), '    "grad_clip": -1.0,'))


SCRIPT = [_edit_weights, _edit_pick, _bad_search, _edit_epochs, _bad_outside, _edit_lr,
          _bad_value, _edit_l2, _bad_import]


def answer(prompt):
    match = re.search(r"# 当前策略（你要修改的程序）\n```python\n(.*?)\n```\n\n# 任务", prompt, re.S)
    code = match.group(1) if match else ""
    with _lock:
        index = _state["calls"]
        _state["calls"] += 1
    for offset in range(len(SCRIPT)):
        result = SCRIPT[(index + offset) % len(SCRIPT)](code)
        if result:
            hypothesis, diff = result
            used = "[mock-001]" if "PLAYBOOK_VERSION: none" not in prompt else "[]"
            return index, f"HYPOTHESIS: {hypothesis}\nUSED_BULLETS: {used}\n\n{diff}\n"
    return index, "HYPOTHESIS: 没有可做的改动\nUSED_BULLETS: []\n"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "mock", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))) or b"{}")
        messages = request.get("messages", [])
        user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")
        index, text = answer(user)
        if _state["log_dir"]:
            Path(_state["log_dir"], f"call_{index:04d}.txt").write_text(
                f"===== SYSTEM =====\n{system}\n\n===== USER =====\n{user}\n\n===== RESPONSE =====\n{text}",
                encoding="utf-8")
        prompt_tokens = (len(system) + len(user)) // 3
        payload = {"id": f"mock-{index}", "object": "chat.completion", "created": int(time.time()),
                   "model": request.get("model", "mock"),
                   "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                   "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": len(text) // 3,
                             "total_tokens": prompt_tokens + len(text) // 3}}
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-dir", help="write every prompt and answer here")
    args = parser.parse_args()
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
        _state["log_dir"] = args.log_dir
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"mock LLM listening on http://{args.host}:{args.port}/v1", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
