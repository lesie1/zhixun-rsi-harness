import copy
import json
from pathlib import Path
import unittest
import uuid

from team_ace.__main__ import bounded_worker, make_config
from team_ace.store import BASE, atomic_text, connect, digest, encoded, exclusive, flush, initialize, location, on_batch, publish, rollback, status
from team_ace.worker import commit_operations, process_one


class AdapterTests(unittest.TestCase):
    def config(self, task='A', mode='fixture'):
        cfg = make_config(task, 'smoke', mode)
        name = self._testMethodName + '-' + uuid.uuid4().hex[:8]
        cfg['ace']['storage_root'] = str(BASE / 'state/tests' / name)
        cfg['work_dir'] = str(BASE / 'runs/tests' / name)
        cfg['playbook_path'] = str(Path(cfg['ace']['storage_root']) / task /
                                  digest(cfg['ace']['scope'])[:16] / 'playbook.md')
        initialize(cfg)
        return cfg

    def record(self, cfg, iteration=0, **updates):
        result = {'schema': 'harness.iteration.v1', 'run_id': 'fixture-test',
            'stage_id': cfg['stage_id'], 'task_name': cfg['task_name'], 'iteration': iteration,
            'kind': 'evaluated', 'status': 'ok', 'outcome': 'equal', 'program_id': str(iteration),
            'hypothesis': '模拟输入，不是实际优化收益。', 'primary': {'name': 'macro_f1' if cfg['task_name'] == 'A' else 'rmse'},
            'used_bullets': [], 'flags': [], 'guard_violations': []}
        result.update(updates)
        return result

    def test_duplicate_and_conflicting_evidence(self):
        cfg = self.config()
        record = self.record(cfg)
        self.assertEqual(on_batch([record], cfg), 1)
        self.assertEqual(on_batch([record, record], cfg), 0)
        with self.assertRaises(ValueError):
            on_batch([dict(record, outcome='improved')], cfg)
        self.assertEqual(status(cfg)['records'], 1)
        self.assertEqual(status(cfg)['jobs'], {'pending': 1})

    def test_cross_task_and_gate_records_rejected(self):
        cfg = self.config()
        for change in ({'task_name': 'B'}, {'phase': 'gate'}, {'gate_decision': {}}, {'kind': 'test'}, {'stage_id': 'other'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                on_batch([self.record(cfg, **change)], cfg)
        self.assertEqual(status(cfg)['records'], 0)

    def test_tail_replay_is_idempotent(self):
        cfg = self.config()
        first, tail = self.record(cfg), self.record(cfg, 1, outcome='regressed')
        on_batch([first], cfg)
        path = Path(cfg['work_dir']) / 'records/iterations.jsonl'
        atomic_text(path, encoded(first) + '\n' + encoded(tail) + '\n')
        self.assertEqual(flush(cfg), 1)
        self.assertEqual(flush(cfg), 0)
        self.assertEqual(status(cfg)['records'], 2)

    def test_abc_upstream_roles_are_isolated_candidates(self):
        folders = []
        for task in 'ABC':
            cfg = self.config(task)
            folders.append(location(cfg))
            on_batch([self.record(cfg, outcome='regressed')], cfg)
            self.assertEqual(process_one(cfg)['status'], 'done')
            db = connect(cfg)
            try:
                entries = [json.loads(row['body']) for row in db.execute('SELECT body FROM entries')]
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0]['task'], task)
                self.assertEqual(entries[0]['status'], 'candidate')
                self.assertFalse(entries[0]['independent_validation'])
                self.assertEqual(entries[0]['outcomes'], {'regressed': 1})
                artifact = json.loads(db.execute('SELECT artifact FROM jobs').fetchone()[0])
                self.assertTrue(all('/environment/vendor/ace/ace/core/' in p for p in artifact['role_files']))
                self.assertEqual(len(artifact['calls']), 2)
                self.assertTrue(all(c['call_info']['usage'] is None for c in artifact['calls']))
            finally:
                db.close()
            self.assertIn('模拟', Path(cfg['playbook_path']).read_text())
            # A live view never imports fixture-generated experience.
            cfg['ace']['mode'] = 'live'
            db = connect(cfg)
            try:
                publish(cfg, db)
            finally:
                db.close()
            self.assertNotIn(entries[0]['id'], Path(cfg['playbook_path']).read_text())
        self.assertEqual(len(set(folders)), 3)

    def test_malformed_reply_preserves_memory_and_retry(self):
        cfg = self.config()
        before = Path(cfg['playbook_path']).read_bytes()
        cfg['ace']['fixture_failure'] = 'malformed'
        on_batch([self.record(cfg)], cfg)
        self.assertEqual(process_one(cfg)['status'], 'failed')
        self.assertEqual(Path(cfg['playbook_path']).read_bytes(), before)
        self.assertEqual(status(cfg)['entries'], 0)
        cfg['ace'].pop('fixture_failure')
        db = connect(cfg)
        with db:
            db.execute("UPDATE jobs SET status='pending' WHERE status='failed'")
        db.close()
        self.assertEqual(process_one(cfg)['status'], 'done')
        self.assertEqual(status(cfg)['records'], 1)
        self.assertEqual(status(cfg)['entries'], 1)

    def test_budget_and_disabled_mode(self):
        cfg = self.config(mode='disabled')
        on_batch([self.record(cfg)], cfg)
        self.assertEqual(process_one(cfg)['status'], 'disabled')
        self.assertEqual(status(cfg)['calls'], {})
        cfg['ace'].update(mode='fixture', max_total_calls=1)
        self.assertEqual(process_one(cfg)['status'], 'failed')
        self.assertEqual(status(cfg)['calls'], {'fixture': 1})
        self.assertEqual(status(cfg)['entries'], 0)

    def test_forged_memory_rejected_atomically(self):
        cfg = self.config()
        on_batch([self.record(cfg)], cfg)
        db = connect(cfg)
        try:
            row = db.execute('SELECT id,body FROM records').fetchone()
            evidence = [{'evidence_id': row['id'], **json.loads(row['body'])}]
            job = db.execute('SELECT id FROM jobs').fetchone()[0]
            proposal = {'task': 'A', 'symptom': 'x', 'action': 'y', 'scope_note': 'z', 'evidence_ids': ['invented']}
            cross = dict(proposal, task='B', action='w')
            # One cross-task proposal rejects the whole batch: nothing of it is committed.
            with self.assertRaises(ValueError):
                commit_operations(db, cfg, job, evidence, [{'type': 'ADD', 'content': encoded(proposal)},
                                                          {'type': 'ADD', 'content': encoded(cross)}], {})
            self.assertEqual(db.execute('SELECT count(*) FROM entries').fetchone()[0], 0)
            # Cited evidence IDs are never trusted (2026-11-24): the job's own evidence is attached instead,
            # so an invented ID cannot enter memory.
            commit_operations(db, cfg, job, evidence, [{'type': 'ADD', 'content': encoded(proposal)}], {})
            stored = json.loads(db.execute('SELECT body FROM entries').fetchone()[0])
            self.assertEqual(stored['evidence_ids'], [row['id']])
        finally:
            db.close()

    def test_versions_repair_and_rollback(self):
        cfg = self.config()
        original = Path(cfg['playbook_path']).read_text()
        import hashlib
        version = hashlib.sha256(original.strip().encode()).hexdigest()[:12]
        on_batch([self.record(cfg)], cfg)
        self.assertEqual(process_one(cfg)['status'], 'done')
        with Path(cfg['playbook_path']).open('w') as stream:
            stream.write('injected torn view')
        self.assertEqual(process_one(cfg)['status'], 'empty')
        self.assertNotIn('torn', Path(cfg['playbook_path']).read_text())
        self.assertEqual(rollback(cfg, version), version)
        self.assertEqual(Path(cfg['playbook_path']).read_text(), original)
        self.assertEqual(status(cfg)['records'], 1)  # Rollback never removes evidence.

    def test_blocked_worker_is_killed_and_evidence_survives(self):
        cfg = self.config()
        on_batch([self.record(cfg)], cfg)
        cfg['ace']['fixture_failure'] = 'hang'
        path = location(cfg) / 'test-config.json'
        atomic_text(path, encoded(cfg))
        result = bounded_worker(path, seconds=2)
        self.assertEqual(result['stop_reason'], 'wall_timeout')
        self.assertEqual(status(cfg)['records'], 1)
        self.assertEqual(status(cfg)['entries'], 0)

    def test_retired_memory_stays_retired_and_operator_lock(self):
        cfg = self.config()
        on_batch([self.record(cfg)], cfg)
        self.assertEqual(process_one(cfg)['status'], 'done')
        db = connect(cfg)
        entry = json.loads(db.execute('SELECT body FROM entries').fetchone()[0])
        entry['status'] = 'retired'
        with db:
            db.execute('UPDATE entries SET body=? WHERE id=?', (encoded(entry), entry['id']))
        db.close()
        on_batch([self.record(cfg, 1)], cfg)
        self.assertEqual(process_one(cfg)['status'], 'done')
        self.assertNotIn(entry['id'], Path(cfg['playbook_path']).read_text())
        db = connect(cfg)
        self.assertEqual(db.execute('SELECT count(*) FROM entries').fetchone()[0], 1)
        db.close()
        with exclusive(cfg):
            self.assertEqual(process_one(cfg)['status'], 'busy')
            with self.assertRaises(RuntimeError):
                rollback(cfg, 'not-a-version')


if __name__ == '__main__':
    unittest.main(verbosity=2)
