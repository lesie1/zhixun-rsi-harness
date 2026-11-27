"""Reuse upstream ACE Reflector/Curator with bounded, replaceable transport."""
import importlib
import json
import os
from pathlib import Path
import re
import sys
import time
import types

from .store import TASKS, atomic_text, encoded

ACE_ROOT = Path('/home/dino/Documents/environment/vendor/ace')


def load_roles():
    # Import only the upstream roles/prompts, not ACE's embedding/analyzer stack.
    # All files remain upstream-owned and unchanged.
    import dotenv
    original = dotenv.load_dotenv
    dotenv.load_dotenv = lambda *a, **k: False  # Never search unrelated .env files.
    if str(ACE_ROOT) not in sys.path:
        sys.path.insert(0, str(ACE_ROOT))
    try:
        for suffix, path in [('', 'ace'), ('.core', 'ace/core'), ('.prompts', 'ace/prompts')]:
            name = '_topic9_upstream_ace' + suffix
            if name not in sys.modules:
                module = types.ModuleType(name)
                module.__path__ = [str(ACE_ROOT / path)]
                sys.modules[name] = module
        reflector = importlib.import_module('_topic9_upstream_ace.core.reflector')
        curator = importlib.import_module('_topic9_upstream_ace.core.curator')
        return reflector, curator
    finally:
        dotenv.load_dotenv = original


def request_limits(ace):
    """Per-call HTTP timeout (seconds) and output-token cap from the ace config.

    Defaults are the original 20 s / 1024 tokens; a reasoning model needs more and the
    caller must set both explicitly (team_sim passes --ace-request-timeout/--ace-max-tokens).
    """
    timeout, tokens = ace.get('request_timeout_s', 20), ace.get('max_tokens', 1024)
    if type(timeout) not in (int, float) or not 5 <= timeout <= 900:
        raise ValueError('request_timeout_s must be within 5..900 seconds')
    if type(tokens) is not int or not 256 <= tokens <= 16384:
        raise ValueError('max_tokens must be within 256..16384')
    return timeout, tokens


def thinking_body(ace):
    """Extra request fields for the provider's thinking switch, or None (nothing extra is sent).

    pytrio's Qwen3.8-27B thinks before it answers; the thinking comes back in a separate field but
    counts against max_tokens, so a long prompt can end with an empty answer. Probed on 2026-11-24:
    {'enable_thinking': False} turns thinking off there, chat_template_kwargs is rejected (HTTP 400)
    and a '/no_think' suffix is ignored. Only sent when ace.enable_thinking is set explicitly.
    """
    if 'enable_thinking' not in ace:
        return None
    if type(ace['enable_thinking']) is not bool:
        raise ValueError('enable_thinking must be true or false')
    return {'enable_thinking': ace['enable_thinking']}


LABELS = (('symptom', ('症状假设', '症状', 'symptom')), ('action', ('建议', '做法', 'action')),
          ('scope_note', ('适用范围', '范围', 'scope_note', 'scope')), ('evidence', ('证据', 'evidence_ids', 'evidence')),
          ('outcomes', ('结果关联', 'outcomes')))
FIELD_OF = {word.lower(): field for field, words in LABELS for word in words}
LABEL = re.compile('(' + '|'.join(sorted((re.escape(w) for _, ws in LABELS for w in ws), key=len, reverse=True))
                   + r')\s*[:：]', re.I)
TASK_MARK = re.compile(r'待验证\s*[｜|]\s*([A-Za-z])\s*[｜|]')


def memory_content(content):
    """The advice inside one ADD operation: {'task' (when stated), 'symptom', 'action', 'scope_note'}.

    Models write it in different ways, and all of them are read:
      - a JSON object, or a string holding one (bare, in a code block, after a sentence); English
        or Chinese keys;
      - one bullet in the playbook's own style, '待验证｜B｜症状假设：…；建议：…；范围：…；证据：…'
        (what Qwen3.8-27B wrote on 2026-11-24, copying the current playbook). Parts are cut at the
        labels, so a '；' inside a part stays;
      - plain advice without labels, taken as the action.
    Evidence IDs and result tallies written by the model are ignored: the worker attaches the job's
    own evidence. Content without any advice raises ValueError with a preview.
    """
    found = content
    if isinstance(content, str):
        found = load_roles()[1].extract_json_from_text(content)
    value = {}
    if isinstance(found, dict):
        for key, item in found.items():
            field = 'task' if str(key).strip().lower() == 'task' else FIELD_OF.get(str(key).strip().lower())
            if field and field not in value and item not in (None, ''):
                value[field] = item
    if 'action' not in value and isinstance(content, str):
        value = {}
        mark = TASK_MARK.search(content)
        if mark:
            value['task'] = mark.group(1)
        labels = list(LABEL.finditer(content))
        for i, label in enumerate(labels):
            end = labels[i + 1].start() if i + 1 < len(labels) else len(content)
            part = content[label.end():end].strip().strip('；;，,。 ').strip()
            field = FIELD_OF[label.group(1).lower()]
            if part and field not in value:
                value[field] = part
        if not labels:
            text = TASK_MARK.sub('', content)
            text = re.sub(r'^\s*[-*]?\s*(\[[^\]]*\])?\s*', '', text).strip()
            if text:
                value['action'] = text
    if not value.get('action'):
        preview = content if isinstance(content, str) else encoded(content)
        raise ValueError(f'ADD content has no usable advice: {str(preview)[:200]!r}')
    return {key: value[key] for key in ('task', 'symptom', 'action', 'scope_note') if key in value}


def strip_reasoning(text):
    """Drop a leading <think>...</think> block that some reasoning models put into the content."""
    if not isinstance(text, str):
        return text
    return re.sub(r'^\s*<think>.*?</think>\s*', '', text, count=1, flags=re.S)


class Transport:
    def __init__(self, db, config, job_id, records):
        self.db, self.config, self.job_id, self.records = db, config, job_id, records
        self.mode = config['ace'].get('mode', 'disabled')
        self.responses = []

    def __call__(self, client, provider, model, prompt, role, call_id, max_tokens=1024, **kwargs):
        ace = self.config['ace']
        if self.mode not in ('fixture', 'live'):
            raise RuntimeError('model transport disabled; evidence stays queued')
        if len(prompt) > 24000:
            raise ValueError('prompt exceeds 24000-character budget; no request sent')
        if self.mode == 'live' and not ace.get('live_calls_approved'):
            raise RuntimeError('live model and budget approval not configured')
        limit = ace.get('max_total_calls', 0)
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError('explicit total call budget in 1..20 required')
        with self.db:
            total = self.db.execute('SELECT count(*) FROM calls WHERE mode=?', (self.mode,)).fetchone()[0]
            if total >= limit:
                raise RuntimeError('model-call budget exhausted')
            row = self.db.execute('INSERT INTO calls(job,role,mode,status,created) VALUES (?,?,?,?,?)',
                (self.job_id, role, self.mode, 'reserved', time.time()))
            row_id = row.lastrowid
        usage, finish = None, None
        try:
            if self.mode == 'fixture':
                response = self.fixture(role)
            else:
                from openai import OpenAI
                endpoint, name = ace.get('api_base'), ace.get('model')
                key_name = ace.get('api_key_env')
                if not endpoint or not name or not key_name or not os.environ.get(key_name):
                    raise ValueError('explicit endpoint, model and designated key environment variable required')
                timeout, token_cap = request_limits(ace)
                with OpenAI(api_key=os.environ[key_name], base_url=endpoint, timeout=timeout, max_retries=0) as client:
                    result = client.chat.completions.create(model=name, temperature=0,
                        messages=[{'role': 'user', 'content': prompt}], max_tokens=min(max_tokens, token_cap),
                        extra_body=thinking_body(ace))
                usage = result.usage.model_dump() if result.usage else None
                finish = result.choices[0].finish_reason
                response = strip_reasoning(result.choices[0].message.content)
                self.keep(row_id, role, prompt, response, finish, usage)
                if not usage:
                    raise ValueError('provider did not report token usage; stop further processing')
                if finish == 'length':
                    raise ValueError(f"reply cut off at the output-token cap ({usage.get('completion_tokens')} "
                                     'tokens); raise max_tokens or turn thinking off')
            self.keep(row_id, role, prompt, response, finish, usage)
            if not isinstance(response, str) or not response.strip() or len(response) > 20000:
                raise ValueError(f'empty or oversized response (finish_reason={finish})')
            info = {'mode': self.mode, 'role': role, 'usage': usage, 'call_id': row_id}
            with self.db:
                self.db.execute('UPDATE calls SET status=?,usage=? WHERE id=?', ('complete', encoded(usage), row_id))
            self.responses.append({'prompt': prompt, 'response': response, 'call_info': info})
            return response, info
        except Exception as error:
            # Keep the token usage of a failed call too: the provider bills it.
            with self.db:
                self.db.execute('UPDATE calls SET status=?,usage=coalesce(?,usage) WHERE id=?',
                    (type(error).__name__, encoded(usage) if usage else None, row_id))
            raise

    def keep(self, row_id, role, prompt, response, finish, usage):
        """Write the prompt and the reply of this call to replies/ beside the scope's database, so the
        exact model text of a failed job can be read afterwards (the job itself keeps it only on success)."""
        if self.config.get('playbook_path'):
            atomic_text(Path(self.config['playbook_path']).parent / 'replies' / f'{row_id:04d}-{role}.json',
                encoded({'job': self.job_id, 'role': role, 'mode': self.mode, 'finish_reason': finish,
                         'usage': usage, 'prompt': prompt, 'response': response}) + '\n')

    def fixture(self, role):
        """Deterministic test replies, never described as a real model's judgement."""
        failure = self.config['ace'].get('fixture_failure')
        if failure == 'timeout':
            raise TimeoutError('injected model timeout')
        if failure == 'hang':
            time.sleep(60)
        if failure == 'malformed':
            return 'not json'
        if role == 'reflector':
            return encoded({'reflection': '模拟反思：仅核对数据与接口；结论待独立实验验证。', 'bullet_tags': []})
        task = self.config['task_name']
        content = {'task': task,
            'symptom': '检查已记录的分类错误' if task == 'A' else '检查已记录的寿命预测误差',
            'action': '先定位错误切片，再做单一改动并与同预算固定续训比较。',
            'scope_note': '仅适用于当前数据及起点；这是模拟回复生成的候选建议。',
            'evidence_ids': [r['evidence_id'] for r in self.records]}
        return encoded({'reasoning': '固定模拟回复，仅测试 ACE 接口。',
            'operations': [{'type': 'ADD', 'section': 'STRATEGIES', 'content': encoded(content)}]})


def experiments(records):
    """Reasoning trace, answer and feedback for upstream's Reflector from the search iterations of one job:
    the generator's hypothesis, its code change and the measured selection-split result."""
    traces, answers, results = [], [], []
    for i, r in enumerate(records, 1):
        tag = f'Experiment {i}: ' if len(records) > 1 else ''
        traces.append(tag + str(r.get('hypothesis') or '(no hypothesis recorded)'))
        answers.append(tag + str(r.get('changes') or '(no code change recorded)'))
        primary = r.get('primary') if isinstance(r.get('primary'), dict) else {}
        values = ', '.join(f'{k}={v:.4g}' for k, v in primary.items()
                           if isinstance(v, (int, float)) and not isinstance(v, bool))
        result = (f"{tag}status={r.get('status')}, outcome={r.get('outcome')}; {primary.get('name', 'metric')}: "
                  f"{values or 'not recorded'}; diagnostic flags: {', '.join(map(str, r.get('flags') or [])) or 'none'}")
        if r.get('guard_violations'):
            result += f"; guard violations: {', '.join(map(str, r['guard_violations']))}"
        if r.get('error'):
            result += f"; error: {str(r['error'])[:300]}"
        results.append(result)
    legend = ('(candidate = model trained with the edited policy; parent_program = trained with the policy it was '
              'edited from; reference_policy = trained with the unedited starting policy; parent_checkpoint = the '
              'starting model before this training.)')
    return '\n'.join(traces), '\n'.join(answers), '\n'.join(results) + '\n' + legend


def propose(db, config, job_id, records, playbook):
    reflector_module, curator_module = load_roles()
    transport = Transport(db, config, job_id, records)
    # Upstream transport retries up to 1000 times. Replace it in this isolated
    # worker only; neither shared source nor other processes are modified.
    reflector_module.timed_llm_call = transport
    curator_module.timed_llm_call = transport
    model = config['ace'].get('model') or 'fixture-not-a-real-model'
    token_cap = request_limits(config['ace'])[1]
    reflector = reflector_module.Reflector(None, 'adapter', model, max_tokens=token_cap)
    curator = curator_module.Curator(None, 'adapter', model, max_tokens=token_cap)
    task = config['task_name']
    metric = next((r['primary']['name'] for r in records if isinstance(r.get('primary'), dict)
                   and r['primary'].get('name')), 'macro_f1' if task == 'A' else 'rmse')
    better = 'higher is better' if metric in ('macro_f1', 'accuracy', 'f1') else 'lower is better'
    # Upstream's Reflector diagnoses "a model's reasoning trace and predicted answer" against
    # environment feedback. Here the model is the search's generator: its hypothesis is the reasoning,
    # its code change the answer, the measured selection-split result the feedback. (Until 2026-11-24
    # this was filled with instructions and "No supervised answer", and Qwen diagnosed a refusal.)
    question = (f"Task {task} ({TASKS[task][0]}; metric {metric}, {better}). A generator model edited the "
        "training-policy code of this task's model (hooks such as the label cap, epochs, learning rate, sample "
        "weights or epoch selection) to improve the metric on a selection split, starting from the current model. "
        "Its hypothesis, its code change and the measured result are given. What does the result teach about this "
        "kind of change for this task? If it helped, why; if it did not, what should be tried instead.")
    trace, answer, feedback = experiments(records)
    reflection, tags, reflection_info = reflector.reflect(question=question, reasoning_trace=trace,
        predicted_answer=answer, ground_truth=None, environment_feedback=feedback, bullets_used=playbook,
        use_ground_truth=False, use_json_mode=True)
    # The Curator template asks for content as one plain bullet and shows the current playbook, so the
    # advice is asked for in the playbook's own labelled style, which memory_content reads.
    context = (question + "\nMeasured result:\n" + feedback + "\n"
        "Write the content of each ADD operation as ONE bullet in Chinese, in the same style as the current "
        "playbook: 症状假设：<the situation in which it applies>；建议：<what to change in the training policy>；"
        "范围：<the data and models it applies to>. Keep each part under 100 characters. The advice is a candidate "
        "to be checked by later experiments, not a proven fact. Do not write IDs or result tallies; the system "
        "attaches the evidence. Only advice about training this task's model, never about prompts, output formats "
        "or how to answer.")
    curator.propose_operations(current_playbook=playbook, recent_reflection=reflection, question_context=context,
        current_step=1, total_samples=len(records), token_budget=800,
        playbook_stats={'entries_are_candidates': True}, use_ground_truth=False, use_json_mode=True)
    # Read the Curator's reply ourselves with upstream's extractor: upstream rejects the whole reply when
    # a field we do not use (reasoning, section) is missing. Non-ADD operations are ignored, as upstream does.
    reply = transport.responses[-1]['response']
    raw = curator_module.extract_json_from_text(reply, 'operations')
    if not isinstance(raw, dict) or not isinstance(raw.get('operations'), list):
        raise ValueError(f'curator reply has no operations list: {reply[:200]!r}')
    operations = [op for op in raw['operations'] if isinstance(op, dict) and str(op.get('type', '')).upper() == 'ADD']
    return operations[:5], {'reflection': reflection, 'untrusted_bullet_tags': tags,
        'ignored_operations': len(raw['operations']) - len(operations),
        'calls': transport.responses, 'upstream': str(ACE_ROOT),
        'role_files': [reflector_module.__file__, curator_module.__file__]}
