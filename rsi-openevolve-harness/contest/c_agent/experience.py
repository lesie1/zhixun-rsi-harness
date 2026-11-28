"""Evidence-backed C action memory. No model calls, code generation, or gate reads."""
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile

from action_space import MENU, validate_settings

SCHEMA = 'C-experience.v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def array_digest(arrays):
    result = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        result.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        result.update(value.tobytes(order='C'))
    return result.hexdigest()


def memory_context(model, train_sha256, selection, initial_sha256, actions_sha256):
    meta = model.meta
    if meta['task'] != 'regression' or meta.get('target_unit') not in ('cycles', None):
        raise ValueError('C experience memory requires engine-RUL regression in cycles')
    domain = {'task': 'C', 'type': meta['task'], 'unit': meta.get('target_unit'),
              'paths': meta['paths'], 'hidden': meta['hidden'],
              'target_mean': meta.get('target_mean'), 'target_scale': meta.get('target_scale'),
              'normalization': array_digest({'mean': model.mean, 'scale': model.scale}),
              'initial_policy': initial_sha256, 'actions_code': actions_sha256}
    parent_arrays = {'mean': model.mean, 'scale': model.scale}
    for prefix, values in [('params', model.params), ('first', model.first), ('second', model.second)]:
        parent_arrays.update({prefix+'.'+name: value for name, value in values.items()})
    keys = ['format', 'task', 'paths', 'hidden', 'seed', 'target_mean', 'target_scale',
            'learning_rate', 'batch_size', 'epochs', 'steps', 'target_unit']
    context = {'parent': digest({'meta': {k: meta.get(k) for k in keys}, 'arrays': array_digest(parent_arrays)}),
               'train': train_sha256,
               'selection': array_digest({'x': selection['x'], 'y': selection['y']}),
               'ids': digest([str(value) for value in selection['ids']])}
    return {'domain': domain, 'context_id': digest(context)}


def conditions(feedback):
    tails = feedback.get('target_slices', [])
    groups = feedback.get('source_subsets', [])
    worst = max(groups, key=lambda value: value['rmse'])['group'] if groups else None
    return {'low_overestimated': bool(tails and tails[0]['bias'] > 0),
            'high_underestimated': bool(tails and tails[-1]['bias'] < 0),
            'worst_source': worst if worst in ('FD001', 'FD002', 'FD003', 'FD004') else None}


def violation_names(decision):
    result = []
    for item in decision.get('violations', [])[:12]:
        name = item.get('guard', '') if isinstance(item, dict) else item
        # Only diagnostic identifiers are retained; model prose never becomes a card.
        if isinstance(name, str) and len(name) <= 100 and all(c.isalnum() or c in ':_+.-' for c in name):
            result.append(name)
    return result


class ExperienceStore:
    def __init__(self, root, namespace, context=None, rules=None, enabled=True, limit=3, max_chars=2600):
        if namespace not in ('live', 'offline'):
            raise ValueError('Experience namespace must be live or offline')
        if (type(enabled) is not bool or type(limit) is not int or type(max_chars) is not int or
                not 1 <= limit <= 8 or not 500 <= max_chars <= 5000):
            raise ValueError('Invalid experience configuration')
        self.enabled, self.namespace, self.context = enabled, namespace, context
        self.limit, self.max_chars = limit, max_chars
        self.root = Path(root).resolve()/namespace
        self.path = self.root/'cards.sqlite3'
        self.domain_id = digest({'schema': SCHEMA, 'domain': context['domain'], 'rules': rules}) if context else None
        self.inserted = self.duplicates = 0
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)
            with self.connection() as db:
                db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)')
                db.execute('INSERT OR IGNORE INTO metadata VALUES (?,?)', ('namespace', namespace))
                if db.execute('SELECT value FROM metadata WHERE key=?', ('namespace',)).fetchone()[0] != namespace:
                    raise ValueError('Experience namespace mismatch')
                db.execute('CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, domain TEXT NOT NULL, card TEXT NOT NULL, context TEXT NOT NULL, payload TEXT NOT NULL, created TEXT NOT NULL)')
                db.execute('CREATE INDEX IF NOT EXISTS domain_events ON events(domain, created)')

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def learn(self, record, before, before_settings, source):
        if not self.enabled:
            return {'status': 'disabled'}
        expected_simulated = self.namespace == 'offline'
        simulated = record.get('simulated_planner')
        provider_simulated = record.get('provider', {}).get('simulated')
        if simulated is not expected_simulated or provider_simulated is not expected_simulated:
            return {'status': 'skipped_provenance'}
        if not record.get('new_training') or record.get('reused'):
            return {'status': 'skipped_no_new_measurement'}
        candidate = record.get('candidate') or {}
        decision = record.get('decision') or {}
        proposal = record.get('proposal') or {}
        if (candidate.get('status') != 'ok' or before.get('status') != 'ok' or
                decision.get('reason') not in ('rmse_improved', 'guard_rejected', 'no_material_rmse_gain') or
                proposal.get('action') not in MENU or proposal.get('action') == 'keep'):
            return {'status': 'skipped_invalid_measurement'}
        scopes = (candidate.get('evaluation_scope'), before.get('evaluation_scope'))
        if any(scopes) and scopes[0] != scopes[1]:
            return {'status': 'skipped_scope_mismatch'}
        for value in (before, candidate):
            if any(not math.isfinite(value['metrics'][key]) or value['metrics'][key] < 0
                   for key in ('rmse', 'mae', 'nasa_score')):
                raise ValueError('Non-finite or negative experience metrics')
        recipe = {'action': proposal['action'],
                  'base_settings': asdict(validate_settings(record['base_settings'])),
                  'candidate_settings': asdict(validate_settings(record['settings'])),
                  'control_kind': 'parent' if before_settings is None else 'trained_settings'}
        card_id = 'CEXP-'+digest({'domain': self.domain_id, 'recipe': recipe})[:20]
        evidence_id = digest([card_id, self.context['context_id']])
        old, new = before['metrics'], candidate['metrics']
        delta = {'rmse_relative_gain': (old['rmse']-new['rmse'])/max(old['rmse'], 1e-12),
                 'nasa_relative_change': new['nasa_score']/old['nasa_score']-1 if old['nasa_score'] else None,
                 'mae_relative_change': new['mae']/old['mae']-1 if old['mae'] else None}
        for label, index in [('low', 0), ('high', -1)]:
            if before.get('target_slices') and candidate.get('target_slices'):
                delta[label+'_bias_delta'] = candidate['target_slices'][index]['bias']-before['target_slices'][index]['bias']
        payload = {'schema': SCHEMA, 'namespace': self.namespace, 'recipe': recipe,
                   'conditions': conditions(before), 'accepted': bool(decision['accepted']),
                   'decision': decision['reason'], 'guard_flags': violation_names(decision),
                   'delta': delta, 'source': source}
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False)
        now = datetime.now(timezone.utc).isoformat()
        with self.connection() as db:
            inserted = db.execute('INSERT OR IGNORE INTO events VALUES (?,?,?,?,?,?)',
                                  (evidence_id, self.domain_id, card_id, self.context['context_id'], encoded, now)).rowcount
        self.inserted += bool(inserted)
        self.duplicates += not inserted
        self.export()
        return {'status': 'learned' if inserted else 'duplicate_ignored', 'card_id': card_id, 'evidence_id': evidence_id}

    def cards(self, all_domains=False):
        if not self.enabled:
            return []
        with self.connection() as db:
            rows = db.execute('SELECT domain,card,context,payload,created FROM events'+
                              ('' if all_domains else ' WHERE domain=?')+' ORDER BY created,id',
                              () if all_domains else (self.domain_id,)).fetchall()
        grouped = {}
        for domain, card_id, context, encoded, created in rows:
            item = json.loads(encoded)
            if item.get('schema') != SCHEMA or item.get('namespace') != self.namespace:
                raise ValueError('Experience schema/namespace mismatch')
            validate_settings(item['recipe']['base_settings'])
            validate_settings(item['recipe']['candidate_settings'])
            if item['recipe']['action'] not in MENU:
                continue
            group = grouped.setdefault(card_id, {'id': card_id, 'domain': domain, 'events': []})
            group['events'].append(dict(item, context=context, created=created))
        return list(grouped.values())

    def brief(self, card):
        events = card['events']
        last = events[-1]
        accepted = sum(value['accepted'] for value in events)
        return {'id': card['id'], **last['recipe'], 'contexts': len(events),
                'accepted': accepted, 'rejected': len(events)-accepted,
                'mixed_evidence': 0 < accepted < len(events), 'last_decision': last['decision'],
                'last_guard_flags': last['guard_flags'], 'last_change': last['delta'],
                'prior_conditions': last['conditions'],
                'same_context_seen': bool(self.context and any(x['context'] == self.context['context_id'] for x in events)),
                'use': 'prior selection evidence; retest before adoption'}

    def retrieve(self, current, settings):
        if not self.enabled:
            return []
        current_conditions = conditions(current)
        base = asdict(settings)
        def score(card):
            value = self.brief(card)
            return (4*int(value['base_settings'] == base)+2*value['same_context_seen']+
                    sum(value['prior_conditions'][key] == item for key, item in current_conditions.items()))
        cards = sorted(self.cards(), key=lambda item: (score(item), item['events'][-1]['created'], item['id']), reverse=True)
        chosen = cards[:self.limit]
        warning = next((c for c in cards if any(not event['accepted'] for event in c['events'])), None)
        if self.limit > 1 and warning and warning not in chosen:
            chosen = chosen[:self.limit-1]+[warning]
        result = []
        for card in chosen:
            item = self.brief(card)
            if len(json.dumps(result+[item], ensure_ascii=False, separators=(',', ':'))) <= self.max_chars:
                result.append(item)
        return result

    def summary(self):
        cards = self.cards(all_domains=True) if self.enabled else []
        return {'enabled': self.enabled, 'namespace': self.namespace, 'database': str(self.path),
                'cards': len(cards), 'events': sum(len(card['events']) for card in cards),
                'domains': len(set(card['domain'] for card in cards)), 'current_domain': self.domain_id,
                'new_events_this_process': self.inserted, 'duplicate_replays_ignored': self.duplicates}

    def export(self):
        data = {'schema': SCHEMA, **self.summary(),
                'cards': [dict(self.brief(c), sources=[dict(event['source'], context_id=event['context'])
                                                      for event in c['events']])
                          for c in self.cards(all_domains=True)]}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.root,
                                             prefix='.cards-', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
                stream.write('\n')
            os.replace(temporary, self.root/'cards.json')
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
