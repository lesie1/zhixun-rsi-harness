"""Durable, task-scoped evidence queue and versioned candidate playbooks."""
import hashlib
import contextlib
import fcntl
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import time

BASE = Path(__file__).resolve().parents[1]
TASKS = {'A': ('classification', 'class'), 'B': ('regression', 'minutes'), 'C': ('regression', 'cycles')}
FIELDS = ('schema', 'run_id', 'stage_id', 'task_name', 'iteration', 'program_id', 'kind', 'status',
          'outcome', 'hypothesis', 'changes', 'primary', 'metrics', 'flags', 'guard_violations',
          'feedback_text', 'error', 'code_sha256', 'parent_code_sha256', 'used_bullets',
          'playbook_version', 'token_usage', 'iteration_time')


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encoded(value).encode()).hexdigest()


def atomic_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix='.ace-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def scope_for(config):
    task = config['task_name']
    if task not in TASKS:
        raise ValueError('only tasks A/B/C are supported')
    hashes = {}
    for key in ('train_path', 'validation_path', 'parent_checkpoint'):
        p = Path(config[key]).resolve()
        h = hashlib.sha256()
        with p.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                h.update(chunk)
        hashes[key] = h.hexdigest()
    for key in ('split_manifest', 'spec_file'):
        if config.get(key):
            hashes[key] = hashlib.sha256(Path(config[key]).read_bytes()).hexdigest()
    return {'task': task, 'task_type': TASKS[task][0], 'unit': TASKS[task][1],
            'source_hashes': hashes, 'split_manifest': config.get('split_manifest'),
            'split_options': config.get('split', {}), 'seed': config.get('seed', 42),
            'spec_file': config.get('spec_file'),
            'protocol': 'selection-feedback-only-v1'}


def location(config):
    ace = config['ace']
    scope = ace['scope']
    task = config['task_name']
    if task not in TASKS or scope['task'] != task:
        raise ValueError('task and scope do not match')
    root = Path(ace['storage_root']).resolve()
    if BASE / 'state' not in root.parents:
        raise ValueError('ACE storage must be below this adapter state/ directory')
    folder = root / task / digest(scope)[:16]
    expected = folder / 'playbook.md'
    if Path(config['playbook_path']).resolve() != expected:
        raise ValueError('playbook path must match task/scope storage')
    return folder


def connect(config):
    folder = location(config)
    folder.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(folder / 'memory.sqlite3', timeout=3)
    db.row_factory = sqlite3.Row
    db.executescript('''
        CREATE TABLE IF NOT EXISTS records(id TEXT PRIMARY KEY, body TEXT NOT NULL, origin TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, ids TEXT NOT NULL, status TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0, error TEXT, artifact TEXT);
        CREATE TABLE IF NOT EXISTS entries(id TEXT PRIMARY KEY, body TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS versions(version TEXT PRIMARY KEY, text TEXT NOT NULL, snapshot TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS calls(id INTEGER PRIMARY KEY, job TEXT, role TEXT, mode TEXT,
            status TEXT, usage TEXT, created REAL);
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
    ''')
    prior = db.execute("SELECT value FROM meta WHERE key='scope'").fetchone()
    if prior and prior['value'] != encoded(config['ace']['scope']):
        db.close()
        raise ValueError('storage scope changed')
    with db:
        db.execute("INSERT OR IGNORE INTO meta VALUES ('scope',?)", (encoded(config['ace']['scope']),))
    return db


def initialize(config):
    db = connect(config)
    try:
        publish(config, db)
    finally:
        db.close()


@contextlib.contextmanager
def exclusive(config):
    """Operator edits must not race an in-flight reflection/curation commit."""
    folder = location(config)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'worker.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('ACE worker is active; retry this edit after it finishes')
        yield


def on_batch(records, config):
    """Fast hook: validate and durably enqueue evidence; no network or reflection."""
    if not 0 <= len(records) <= 100:
        raise ValueError('record batch too large')
    accepted = []
    for record in records:
        if record.get('schema') != 'harness.iteration.v1':
            raise ValueError('unknown record schema')
        if record.get('task_name') != config['task_name'] or record.get('stage_id') != config['stage_id']:
            raise ValueError('cross-task or cross-stage record rejected')
        if record.get('kind') not in ('evaluated', 'failed', 'db_rejected'):
            raise ValueError('only search iteration feedback is accepted')
        if record.get('phase', 'selection') != 'selection' or any(k in record for k in ('gate', 'test_metrics', 'gate_decision')):
            raise ValueError('gate/test feedback must not enter the learning loop')
        if not record.get('run_id') or type(record.get('iteration')) is not int:
            raise ValueError('run_id and integer iteration are required')
        body = {k: record.get(k) for k in FIELDS}
        # Store reasoning as a hypothesis; do not promote it into measured evidence.
        for key, limit in [('feedback_text', 5000), ('hypothesis', 500), ('error', 1500), ('changes', 1500)]:
            if body[key] is not None:
                body[key] = str(body[key])[:limit]
        text = encoded(body)
        if len(text.encode()) > 32000:
            raise ValueError('record exceeds bounded evidence size')
        identity = digest([body[k] for k in ('run_id', 'stage_id', 'task_name', 'iteration', 'program_id', 'kind')])
        accepted.append((identity, text))
    new_ids = []
    db = connect(config)
    try:
        with db:
            for identity, text in accepted:
                prior = db.execute('SELECT body FROM records WHERE id=?', (identity,)).fetchone()
                if prior:
                    if prior['body'] != text:
                        raise ValueError('same experiment identity has conflicting evidence')
                    continue
                db.execute('INSERT INTO records VALUES (?,?,?)',
                           (identity, text, config['ace'].get('record_origin', 'unverified_import')))
                new_ids.append(identity)
            if new_ids:
                job = digest(sorted(new_ids))
                db.execute('INSERT INTO jobs(id,ids,status) VALUES (?,?,?)', (job, encoded(new_ids), 'pending'))
    finally:
        db.close()
    return len(new_ids)


def flush(config):
    """Replay all durable records, including the harness's unflushed tail; idempotent."""
    source = Path(config['work_dir']) / 'records' / 'iterations.jsonl'
    if not source.is_file():
        return 0
    count, batch = 0, []
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.endswith('\n'):
                break  # The writer may not have completed the last record yet.
            if line.strip():
                batch.append(json.loads(line))
            if len(batch) == 10:
                count += on_batch(batch, config)
                batch = []
    if batch:
        count += on_batch(batch, config)
    return count


def render_settings(ace):
    """None (default: entries in ID order up to 1900 characters) or the checked ace.render settings.

    ace.render = {'base_first': true, 'max_learned': N, 'max_chars': C} (team_sim sets it): preset
    base experiences first, then at most N learned entries, newest first, up to C characters. Learned
    IDs are hashes, so in ID order they came before the preset 'R-00-BASE-...' entries and, as more
    were carried from round to round, pushed them out of the 1900-character view in hash order.
    """
    render = ace.get('render')
    if not render:
        return None
    learned, chars = render.get('max_learned', 3), render.get('max_chars', 1900)
    if render.get('base_first') is not True or type(learned) is not int or not 0 <= learned <= 20 \
            or type(chars) is not int or not 500 <= chars <= 4000:
        raise ValueError('ace.render needs base_first true, max_learned 0..20 and max_chars 500..4000')
    return {'max_learned': learned, 'max_chars': chars}


def publish(config, db):
    """Render a bounded view from authoritative DB state; repairable after interruption."""
    task = config['task_name']
    mode = config['ace'].get('mode', 'disabled')
    render = render_settings(config['ace'])
    limit = render['max_chars'] if render else 1900
    lines = [f'# {task} 任务候选经验', '以下均为待验证建议，不是已证明有效的技能。']
    if mode == 'fixture':
        lines.append('【模拟回复接口测试；不得作为真实模型效果证据】')
    entries = [json.loads(row['body']) for row in db.execute('SELECT body FROM entries ORDER BY id')]
    shown = [entry for entry in entries
             if entry['status'] == 'candidate' and entry['scope_id'] == digest(config['ace']['scope'])
             and not (entry['mode'] == 'fixture' and mode != 'fixture')]
    if render:
        learned = sorted((e for e in shown if not e.get('base_experience')),
                         key=lambda e: (-float(e.get('updated') or 0), e['id']))
        shown = [e for e in shown if e.get('base_experience')] + learned[:render['max_learned']]
    exposed = []
    for entry in shown:
        text = (f"- [{entry['id']}] 待验证｜{task}｜症状假设：{entry['symptom']}；"
                f"建议：{entry['action']}；范围：{entry['scope_note']}；"
                f"证据：{','.join(x[:10] for x in entry['evidence_ids'])}；"
                f"结果关联：{entry['outcomes']}（不代表因果）")
        if len('\n'.join(lines + [text])) > limit:
            break
        lines.append(text)
        exposed.append(entry['id'])
    if not exposed:
        lines.append('暂无可用候选经验。')
    text = '\n'.join(lines).strip() + '\n'
    version = hashlib.sha256(text.strip().encode()).hexdigest()[:12]
    with db:
        db.execute('INSERT OR IGNORE INTO versions VALUES (?,?,?)', (version, text, encoded(entries)))
    folder = location(config)
    atomic_text(folder / 'versions' / (version + '.md'), text)
    atomic_text(config['playbook_path'], text)
    atomic_text(folder / 'view.json', encoded({'version': version, 'exposed_ids': exposed,
                 'scope': config['ace']['scope'], 'entry_count': len(entries)}) + '\n')
    return version


def rollback(config, version):
    with exclusive(config):
        db = connect(config)
        try:
            row = db.execute('SELECT snapshot FROM versions WHERE version=?', (version,)).fetchone()
            if not row:
                raise ValueError('unknown version in this task/scope')
            entries = json.loads(row['snapshot'])
            with db:
                db.execute('DELETE FROM entries')
                for entry in entries:
                    db.execute('INSERT INTO entries VALUES (?,?)', (entry['id'], encoded(entry)))
            return publish(config, db)
        finally:
            db.close()


def status(config):
    db = connect(config)
    try:
        return {'task': config['task_name'], 'folder': str(location(config)),
                'records': db.execute('SELECT count(*) FROM records').fetchone()[0],
                'entries': db.execute('SELECT count(*) FROM entries').fetchone()[0],
                'jobs': dict(db.execute('SELECT status,count(*) FROM jobs GROUP BY status').fetchall()),
                'calls': dict(db.execute('SELECT mode,count(*) FROM calls GROUP BY mode').fetchall())}
    finally:
        db.close()
