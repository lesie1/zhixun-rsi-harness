"""One bounded queue job per subprocess; durable evidence survives errors."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import time

from .backend import memory_content, propose
from .store import atomic_text, connect, digest, encoded, location, publish, scope_for

DEFAULTS = {'symptom': '（模型未单独写出适用的现象）', 'scope_note': '（模型未注明范围）仅限当前任务与数据，待后续实验验证'}


def task_letter(value):
    """'B', 'b', 'Task B', 'B (轴承)' -> 'B'; anything without a leading task letter -> None."""
    match = re.match(r'\s*(?:task\s*)?([ABC])(?![A-Za-z])', str(value), re.I)
    return match.group(1).upper() if match else None


def text_field(value, key):
    value = DEFAULTS.get(key, '') if value in (None, '') else value
    value = ' '.join((value if isinstance(value, str) else encoded(value)).split())
    return value if len(value) <= 250 else value[:249] + '…'


def commit_operations(db, config, job_id, evidence, operations, artifact):
    if len(operations) > 5:
        raise ValueError('at most five memory proposals per job')
    # Provenance is attached by the system (as upstream ACE assigns bullet IDs): an entry made by this
    # job rests on this job's evidence. IDs the model writes are never trusted or stored, so a cited
    # but invented ID cannot enter memory, and a 27B model need not copy 64-character IDs.
    ids = sorted(r['evidence_id'] for r in evidence)
    if not ids:
        raise ValueError('evidence IDs required')
    entries, skipped, now = [], [], time.time()
    for op in operations:
        if op.get('type') != 'ADD':
            raise ValueError('only structured ADD proposals accepted')
        try:
            value = memory_content(op.get('content'))
        except ValueError as error:      # an operation without advice is skipped, not stored
            skipped.append(str(error))
            continue
        stated = task_letter(value['task']) if value.get('task') is not None else None
        if stated and stated != config['task_name']:
            raise ValueError(f'cross-task memory rejected (content says task {stated})')
        value = {'task': config['task_name'],
                 **{key: text_field(value.get(key), key) for key in ('symptom', 'action', 'scope_note')}}
        identity = config['task_name'] + '-' + digest([config['ace']['scope'], config['ace']['mode'], value['symptom'], value['action']])[:12]
        prior = db.execute('SELECT body FROM entries WHERE id=?', (identity,)).fetchone()
        all_ids = sorted(set(ids) | (set(json.loads(prior['body'])['evidence_ids']) if prior else set()))
        outcomes = {}
        for evidence_id in all_ids:
            row = db.execute('SELECT body FROM records WHERE id=?', (evidence_id,)).fetchone()
            result = json.loads(row['body']).get('outcome', 'unknown')
            outcomes[result] = outcomes.get(result, 0) + 1
        # 'updated' orders learned entries newest first when the playbook is rendered with ace.render;
        # an entry proposed again from new evidence counts as new.
        entries.append({'id': identity, **value, 'evidence_ids': all_ids, 'outcomes': outcomes,
            'scope_id': digest(config['ace']['scope']),
            'status': json.loads(prior['body'])['status'] if prior else 'candidate',
            'mode': config['ace']['mode'], 'independent_validation': False,
            'created': (json.loads(prior['body']).get('created') if prior else None) or now, 'updated': now})
    if skipped and not entries:
        raise ValueError(f'no usable advice in the curator operations: {skipped[0]}')
    if skipped:
        artifact = dict(artifact, skipped_operations=skipped)
    # All proposals validate before any of them are committed.
    with db:
        for entry in entries:
            db.execute('INSERT OR REPLACE INTO entries VALUES (?,?)', (entry['id'], encoded(entry)))
        db.execute('UPDATE jobs SET status=?,error=NULL,artifact=? WHERE id=?', ('done', encoded(artifact), job_id))
    publish(config, db)


def process_one(config):
    if config['ace'].get('mode', 'disabled') == 'disabled':
        return {'status': 'disabled', 'model_calls': 0}
    if scope_for(config) != config['ace']['scope']:
        raise ValueError('data/checkpoint/protocol changed; create a fresh scoped config')
    folder = location(config)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'worker.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {'status': 'busy'}
        db = connect(config)
        try:
            publish(config, db)  # Repair a commit-before-render interruption.
            job = db.execute("SELECT * FROM jobs WHERE status='pending' ORDER BY rowid LIMIT 1").fetchone()
            if job is None:
                return {'status': 'empty'}
            if job['attempts'] >= 2:
                with db:
                    db.execute("UPDATE jobs SET status='failed',error='retry limit exhausted' WHERE id=?", (job['id'],))
                return {'status': 'failed', 'job': job['id']}
            with db:
                db.execute('UPDATE jobs SET attempts=attempts+1 WHERE id=?', (job['id'],))
            evidence = []
            for identity in json.loads(job['ids']):
                row = db.execute('SELECT * FROM records WHERE id=?', (identity,)).fetchone()
                record = json.loads(row['body'])
                # Only bounded, selection-side facts and explicitly labelled hypotheses.
                evidence.append({'evidence_id': identity, 'record_origin': row['origin'],
                    **{k: record.get(k) for k in ('task_name', 'status', 'outcome', 'primary',
                       'flags', 'guard_violations', 'hypothesis', 'changes', 'error')}})
            try:
                book = Path(config['playbook_path']).read_text(encoding='utf-8')
                operations, artifact = propose(db, config, job['id'], evidence, book)
                commit_operations(db, config, job['id'], evidence, operations, artifact)
                return {'status': 'done', 'job': job['id'], 'proposals': len(operations)}
            except Exception as error:
                # Preserve queue/evidence; retry is explicit, never automatic or unbounded.
                with db:
                    db.execute("UPDATE jobs SET status='failed',error=? WHERE id=?",
                               (f'{type(error).__name__}: {error}'[:1500], job['id']))
                return {'status': 'failed', 'job': job['id'], 'error': type(error).__name__}
        finally:
            db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    print(encoded(process_one(config)))
