"""Explicit live transport and offline fixtures. No API call at import/check time."""
import hashlib
import json
import os
from pathlib import Path
import time

from backend import write_json
from feedback import SYSTEM


def configured_api_key():
    key = os.environ.get('OPENEVOLVE_API_KEY')
    if not key:
        location = os.environ.get('C_API_KEYS_FILE')
        path = Path(location) if location else Path(__file__).resolve().parents[1]/'secrets/api_keys.json'
        if not path.is_file():
            raise ValueError('Set OPENEVOLVE_API_KEY or C_API_KEYS_FILE before live planning')
        key = json.loads(path.read_text(encoding='utf-8-sig')).get('OPENEVOLVE_API_KEY')
    if (not isinstance(key, str) or not key or key.startswith('REPLACE_') or
            any(character.isspace() for character in key)):
        raise ValueError('Missing or invalid configured API key')
    return key


class BudgetExhausted(RuntimeError):
    pass


def messages_for(observation):
    return [{'role': 'system', 'content': SYSTEM},
            {'role': 'user', 'content': json.dumps(observation, ensure_ascii=False, separators=(',', ':'))}]


class FixturePlanner:
    simulated = True

    def __init__(self):
        self.prompts = []

    def propose(self, observation, turn):
        self.prompts.append(observation)
        actions = ['pick_best_epoch', 'epochs_1', 'epochs_3',
                   'tail_error_1_1', 'change_architecture', 'keep']
        action = actions[(turn - 1) % len(actions)]
        if action == 'high_rul_weight_1_2':
            high = observation['incumbent']['target_slices'][-1]
            reason = f"离线测试：高寿命切片偏差{high['bias']:.3f}，测试轻度加权接口"
        elif turn > 1:
            last = observation['previous_trials'][-1]
            reason = f"离线测试：已读到{last['id']}的{last['decision']['reason']}，检查后续动作"
        else:
            reason = '离线测试：检查最佳epoch动作；这不是Qwen生成的实验'
        return json.dumps({'action': action, 'why': reason, 'used': observation['available_evidence_ids'][-1:]},
                          ensure_ascii=False), {'simulated': True, 'real_api_calls': 0}


class LivePlanner:
    simulated = False

    def __init__(self, run, max_calls, max_total_tokens=30000, max_output_tokens=600,
                 timeout=60, seed=42, client_factory=None):
        self.run = Path(run)
        self.max_calls = max_calls
        self.max_total_tokens = max_total_tokens
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self.seed = seed
        self.client_factory = client_factory
        self.ledger_path = self.run / 'provider-ledger.json'
        if self.ledger_path.exists():
            raise ValueError('Existing provider budget cannot be reset or silently replayed')
        self.ledger = {}

    def propose(self, observation, turn):
        messages = messages_for(observation)
        encoded = json.dumps(messages, ensure_ascii=False).encode('utf-8')
        reserve = len(encoded) + self.max_output_tokens + 1024
        if len(encoded) > 40000:
            raise BudgetExhausted('Prompt exceeds local byte limit')
        if (len(self.ledger) >= self.max_calls or str(turn) in self.ledger or
                sum(r['charged'] for r in self.ledger.values()) + reserve > self.max_total_tokens):
            raise BudgetExhausted('Call/token budget reached; no request was sent')
        entry = {'status': 'reserved', 'charged': reserve, 'prompt_sha256': hashlib.sha256(encoded).hexdigest()}
        self.ledger[str(turn)] = entry
        write_json(self.ledger_path, self.ledger)
        started = time.monotonic()
        record = {'mode': 'live', 'turn': turn, 'messages': messages, 'max_output_tokens': self.max_output_tokens,
                  'transport_simulated': self.client_factory is not None}
        try:
            if self.client_factory:
                client = self.client_factory()
            else:
                from openai import OpenAI
                key = configured_api_key()
                client = OpenAI(api_key=key, base_url='https://pytrio.com/api/openai/v1',
                                timeout=self.timeout, max_retries=0)
            with client:
                reply = client.chat.completions.create(
                    model='Qwen/Qwen3.8-27B', messages=messages, temperature=0.25, seed=self.seed + turn,
                    max_tokens=self.max_output_tokens, response_format={'type': 'json_object'},
                    extra_body={'enable_thinking': False})
            usage = reply.usage.model_dump() if reply.usage else {}
            actual = usage.get('total_tokens')
            if type(actual) is int and actual >= 0:
                entry['charged'] = actual
            choice = reply.choices[0]
            record.update(raw=choice.message.content, usage=usage, finish_reason=choice.finish_reason,
                          returned_model=reply.model, provider_id=reply.id)
            if reply.model != 'Qwen/Qwen3.8-27B' or choice.finish_reason != 'stop' or not isinstance(choice.message.content, str):
                raise ValueError('Incomplete or mismatched model response')
            entry['status'] = 'ok'
            return choice.message.content, {'simulated': self.client_factory is not None, 'usage': usage,
                                            'finish_reason': choice.finish_reason}
        except Exception as error:
            entry['status'] = type(error).__name__
            # Exception messages can contain provider internals; preserve only safe metadata.
            record.update(error_type=type(error).__name__, http_status=getattr(error, 'status_code', None))
            raise RuntimeError('Provider request failed: ' + type(error).__name__) from None
        finally:
            record['seconds'] = time.monotonic() - started
            record['status'] = entry['status']
            write_json(self.run / 'calls' / f'{turn:03d}.json', record)
            write_json(self.ledger_path, self.ledger)
