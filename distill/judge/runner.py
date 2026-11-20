"""Independent GPT-5 rejudge; originals are read-only, attempts are durable."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import signal
import threading
import time
import urllib.error
import urllib.request


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def parse_score(content):
    # Ignore any explicitly delimited reasoning; never select an arbitrary digit.
    content = content.rsplit('</think>', 1)[-1]
    matches = re.findall(r'(?im)^\s*(?:\*\*)?Final\s+Score\s*(?:\*\*)?\s*:\s*(?:\*\*)?\s*([0-5])\s*(?:\*\*)?\s*\.?\s*$', content)
    if len(matches) == 1:
        return int(matches[0]), 'explicit_final_score'
    if re.fullmatch(r'\s*[0-5]\s*', content):
        return int(content.strip()), 'bare_integer'
    raise ValueError('missing or ambiguous explicit Final Score')


def parse_response(row, response, model):
    if response.get('model') != model:
        raise ValueError('model mismatch')
    choice = response['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('incomplete response')
    content = choice['message']['content']
    if not isinstance(content, str):
        raise ValueError('missing response text')
    if row['cohort'] == 'egolifeqa':
        verdict = json.loads(content)
        if (set(verdict) != {'correct', 'reason'} or type(verdict['correct']) is not bool
                or not isinstance(verdict['reason'], str)):
            raise ValueError('invalid binary verdict')
        return {'score': int(verdict['correct']), 'raw_score': verdict['correct'], 'reason': verdict['reason']}
    score, path = parse_score(content)
    return {'score': 1 if score >= 4 else .5 if score == 3 else 0, 'raw_score': score, 'parse_path': path}


def usage_cost(response, policy):
    usage = response.get('usage', {})
    if not all(type(usage.get(k)) is int and usage[k] >= 0 for k in ('prompt_tokens', 'completion_tokens')):
        return None
    return (usage['prompt_tokens'] * policy['usd_per_million_input']
            + usage['completion_tokens'] * policy['usd_per_million_output']) / 1e6


class Runner:
    def __init__(self, root, inputs_sha, transport=None, key=None):
        self.root = Path(root)
        raw = (self.root / 'inputs.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != inputs_sha:
            raise ValueError('inputs hash mismatch')
        self.inputs = json.loads(raw)
        self.policy = self.inputs['policy']
        self.policy_sha = digest(self.policy)
        self.rows = {r['id']: r for r in self.inputs['rows'] + self.inputs['controls']}
        self.records = {k: [] for k in self.rows}
        self.lock = threading.RLock()
        self.halt = threading.Event()
        self.pause_until = 0
        self.stop_reason = None
        self.transport = transport or self.http
        self.key = key
        self.started = time.time()
        (self.root / 'attempts').mkdir(exist_ok=True)
        self.load_records()

    def identity(self, row):
        return digest({'id': row['id'], 'answer_sha256': row.get('answer_sha256'),
                       'messages': row['messages'], **{k: self.policy.get(k) for k in
                       ('endpoint', 'model', 'reasoning_effort', 'max_completion_tokens', 'provider_preferences')}})

    def load_records(self):
        known = {self.identity(row) for row in self.rows.values()}
        if any(p.is_dir() and p.name not in known for p in (self.root / 'attempts').iterdir()):
            raise ValueError('unknown attempt identity; refuse to hide earlier spend')
        for row in self.rows.values():
            folder = self.root / 'attempts' / self.identity(row)
            for p in sorted(folder.glob('*.json')):
                rec = json.loads(p.read_text())
                if rec['identity'] != self.identity(row) or rec['id'] != row['id']:
                    raise ValueError('ledger identity mismatch')
                raw_path = folder / (p.stem + '.raw')
                if rec['state'] == 'in_flight' and raw_path.exists():
                    wire = json.loads(raw_path.read_text())
                    self.classify(row, rec, wire)
                    rec['raw_sha256'] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
                    save(p, rec)
                elif rec['state'] == 'in_flight':
                    rec['state'] = 'uncertain_billed'
                    rec['error'] = 'orphaned request; original response unavailable'
                    save(p, rec)
                self.records[row['id']].append(rec)

    def cost(self):
        return sum(r.get('cost_usd', r['reserved_usd']) for records in self.records.values() for r in records)

    def result(self, row):
        if row.get('messages') is None:
            return {'score': 0, 'state': 'generation_timeout_zero'}
        for rec in self.records[row['id']]:
            if rec['state'] == 'valid':
                return {**rec['verdict'], 'state': 'graded', 'attempt': rec['number']}
        return None

    def sample_count(self, row):
        return sum(r['state'] != 'not_sampled' for r in self.records[row['id']])

    def reserve(self, row):
        with self.lock:
            if self.halt.is_set() or self.result(row) is not None:
                return None
            if self.sample_count(row) >= self.policy['max_sample_attempts']:
                return None
            if sum(r['state'] == 'not_sampled' for records in self.records.values() for r in records) >= self.policy['max_rate_limit_records']:
                self.stop_reason = 'rate_limit_record_cap'
                self.halt.set()
                return None
            reserve = (self.policy['max_completion_tokens'] * self.policy['usd_per_million_output']
                       + (len(json.dumps(row['messages'], ensure_ascii=False).encode()) + 2048)
                       * self.policy['usd_per_million_input']) / 1e6
            if self.cost() + reserve > self.policy['spend_ceiling_usd']:
                self.stop_reason = 'operator_spend_ceiling'
                self.halt.set()
                return None
            records = self.records[row['id']]
            number = len(records) + 1
            folder = self.root / 'attempts' / self.identity(row)
            folder.mkdir(exist_ok=True)
            path = folder / ('%04d.json' % number)
            rec = {'id': row['id'], 'identity': self.identity(row), 'number': number,
                   'state': 'in_flight', 'started_unix': time.time(), 'reserved_usd': reserve}
            slot = path.with_suffix('.slot')
            save(slot, rec)
            os.link(slot, path)
            slot.unlink()
            fd = os.open(str(folder), os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            records.append(rec)
            return rec, path

    def http(self, body):
        request = urllib.request.Request(self.policy['endpoint'], data=json.dumps(body).encode(),
                                        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self.key})
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            response = opener.open(request, timeout=self.policy['timeout_seconds'])
        except urllib.error.HTTPError as error:
            response = error
        with response:
            text = response.read().decode('utf-8', errors='replace')
            return {'status': response.status, 'body': text.replace(self.key, '[REDACTED]'),
                    'headers': {k.lower(): v for k, v in response.headers.items()
                                if k.lower() in ('x-request-id', 'openai-processing-ms', 'retry-after')
                                or k.lower().startswith('x-ratelimit-')}}

    def classify(self, row, rec, wire):
        rec['finished_unix'] = time.time()
        rec['http_status'] = wire['status']
        try:
            response = json.loads(wire['body'])
        except (ValueError, TypeError):
            response = {}
        cost = usage_cost(response, self.policy)
        if cost is not None:
            rec['cost_usd'] = cost
            rec['usage'] = response['usage']
        rec['model'] = response.get('model')
        rec['system_fingerprint'] = response.get('system_fingerprint')
        rec['service_tier'] = response.get('service_tier')
        error = response.get('error', {})
        code = error.get('code') if isinstance(error, dict) else None
        if wire['status'] == 429 and code == 'rate_limit_exceeded':
            rec.update(state='not_sampled', cost_usd=0)
            try:
                retry = max(5, min(300, float(wire.get('headers', {}).get('retry-after', '30'))))
            except ValueError:
                retry = 30
            self.pause_until = max(self.pause_until, time.time() + retry)
            return
        if (wire['status'] in (400, 401, 402, 403) or code in ('insufficient_quota', 'credit_balance_exhausted')
                or isinstance(error, dict) and error.get('type') == 'insufficient_quota'):
            rec.update(state='provider_rejected', cost_usd=0, error_code=code)
            self.stop_reason = 'provider_rejected_' + str(wire['status']) + '_' + str(code)
            self.halt.set()
            return
        if wire['status'] != 200:
            rec['state'] = 'uncertain_billed'
            return
        expected = self.policy.get('response_model', self.policy['model'])
        if response.get('model') != expected or (self.policy.get('expected_provider')
                                                 and response.get('provider') != self.policy['expected_provider']):
            rec['state'] = 'identity_failure'
            self.stop_reason = 'model_or_provider_identity_failure'
            self.halt.set()
            return
        rec['provider'] = response.get('provider')
        try:
            rec['verdict'] = parse_response(row, response, expected)
            rec['state'] = 'valid'
        except (ValueError, TypeError, KeyError, IndexError):
            rec['state'] = 'format_or_identity_failure'

    def judge(self, row):
        while not self.halt.is_set():
            if self.halt.wait(max(0, self.pause_until - time.time())):
                return
            slot = self.reserve(row)
            if slot is None:
                return
            rec, path = slot
            if self.policy['provider'] == 'openrouter':
                body = {'model': self.policy['model'], 'reasoning': {'effort': self.policy['reasoning_effort']},
                        'max_tokens': self.policy['max_completion_tokens'], 'provider': self.policy['provider_preferences']}
            else:
                body = {k: self.policy[k] for k in ('model', 'reasoning_effort', 'max_completion_tokens')}
            body['messages'] = row['messages']
            try:
                wire = self.transport(body)
            except (OSError, TimeoutError, http.client.HTTPException):
                with self.lock:
                    rec.update(state='uncertain_billed', finished_unix=time.time(), error='transport failure; billing uncertain')
                    save(path, rec)
                self.halt.wait(5)
                continue
            try:
                save(path.with_suffix('.raw'), wire)
            except OSError:
                self.stop_reason = 'raw_persistence_failed'
                self.halt.set()
                return
            with self.lock:
                self.classify(row, rec, wire)
                rec['raw_sha256'] = hashlib.sha256(path.with_suffix('.raw').read_bytes()).hexdigest()
                save(path, rec)
            if self.result(row) is not None:
                return
            self.halt.wait(5)

    def verify_sources(self):
        changed = [p for p, sha in self.inputs['source_files'].items()
                   if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != sha]
        result = {'checked': len(self.inputs['source_files']), 'unchanged': not changed, 'changed': changed, 'at_unix': time.time()}
        save(self.root / 'source-verification.json', result)
        if changed:
            raise ValueError('original source changed')
        return result

    def verify_raw(self):
        checked = 0
        for row in self.rows.values():
            folder = self.root / 'attempts' / self.identity(row)
            for rec in self.records[row['id']]:
                p = folder / ('%04d.raw' % rec['number'])
                if not p.exists():
                    if rec['state'] == 'valid':
                        raise ValueError('valid record missing raw response')
                    continue
                if hashlib.sha256(p.read_bytes()).hexdigest() != rec['raw_sha256']:
                    raise ValueError('raw response changed')
                copy = dict(rec)
                self.classify(row, copy, json.loads(p.read_text()))
                if copy['state'] != rec['state'] or copy.get('verdict') != rec.get('verdict'):
                    raise ValueError('raw reparse differs')
                checked += 1
        result = {'checked': checked, 'all_agreed': True, 'at_unix': time.time()}
        save(self.root / 'raw-verification.json', result)
        return result

    def report(self):
        with self.lock:
            rows = []
            table = []
            for r in self.inputs['rows']:
                result = self.result(r)
                state = 'judge_unresolved' if self.sample_count(r) >= self.policy['max_sample_attempts'] else 'pending'
                rows.append({k: v for k, v in r.items() if k != 'messages'} | (result or {'state': state, 'score': None}))
            for cohort in ('egolifeqa', 'week', 'day', 'month'):
                group = [r for r in rows if r['cohort'] == cohort]
                if not group:
                    continue
                done = [r for r in group if r['score'] is not None]
                complete = len(done) == len(group)
                table.append({'cohort': cohort, 'total': len(group), 'resolved': len(done),
                              'deepseek_percent': 100 * sum(r['deepseek_score'] for r in group) / len(group),
                              'gpt5_percent': 100 * sum(r['score'] for r in done) / len(group) if complete else None,
                              'gpt5_completed_subset_percent': 100 * sum(r['score'] for r in done) / len(done) if done else None,
                              'paired_deepseek_completed_subset_percent': 100 * sum(r['deepseek_score'] for r in done) / len(done) if done else None,
                              'disagreements': sum(r['score'] != r['deepseek_score'] for r in done),
                              'generation_failed_zero': sum(r['state'] == 'generation_timeout_zero' for r in group),
                              'official_exact_choice_percent': 100 * sum(r['exact_score'] for r in group) / len(group) if cohort == 'egolifeqa' else None,
                              'unresolved': sum(r['state'] == 'judge_unresolved' for r in group)})
            attempts = [r for group in self.records.values() for r in group]
            output = {'at_unix': time.time(), 'policy_sha256': self.policy_sha, 'model': self.policy['model'],
                      'all_completed': all(r['score'] is not None for r in rows), 'arm': self.policy['arm'],
                      'base_rejudged': self.policy['base_rejudged'],
                      'stop_reason': self.stop_reason, 'table': table, 'rows': rows,
                      'estimated_cost_usd_conservative_input_rate': self.cost(),
                      'cost_is_provider_bill': False, 'attempt_records': len(attempts),
                      'attempt_states': {s: sum(r['state'] == s for r in attempts) for s in sorted({r['state'] for r in attempts})},
                      'total_prompt_tokens': sum(r.get('usage', {}).get('prompt_tokens', 0) for r in attempts),
                      'total_completion_tokens': sum(r.get('usage', {}).get('completion_tokens', 0) for r in attempts),
                      'limitations': ['Existing generations only; Ego inputs are selected-option text, not freeform generation.',
                                     'One sampled grade; differences combine judge choice with sampling noise.',
                                     'MM mapping 0/1/2=0, 3=0.5, 4/5=1; Ego binary semantic correctness.']}
            save(self.root / 'progress.json', output)
            return output

    def parallel(self, rows):
        pending = iter(rows)
        last_report = 0
        with ThreadPoolExecutor(max_workers=self.policy['workers']) as pool:
            active = set()
            while not self.halt.is_set() or active:
                while len(active) < self.policy['workers'] and not self.halt.is_set():
                    try:
                        row = next(pending)
                    except StopIteration:
                        break
                    if self.result(row) is None:
                        active.add(pool.submit(self.judge, row))
                if not active:
                    break
                finished = {f for f in active if f.done()}
                for f in finished:
                    f.result()
                active -= finished
                if time.time() - last_report >= 30 or not active:
                    self.report()
                    last_report = time.time()
                if active:
                    time.sleep(5)

    def run(self):
        self.verify_sources()
        if self.policy['provider'] == 'openrouter':
            with urllib.request.urlopen('https://openrouter.ai/api/v1/models', timeout=30) as response:
                catalog = json.load(response)
            model = next(m for m in catalog['data'] if m['id'] == self.policy['model'])
            if model['canonical_slug'] != self.policy['canonical_slug']:
                raise ValueError('model catalog snapshot differs')
            save(self.root / 'model-catalog.json', model)
        self.parallel(self.inputs['controls'])
        controls = []
        for r in self.inputs['controls']:
            result = self.result(r)
            expected = r['expected']
            passed = result is not None and (result['raw_score'] in expected if isinstance(expected, list)
                                             else result['raw_score'] is expected)
            controls.append({'id': r['id'], 'passed': passed, 'result': result})
        save(self.root / 'controls.json', controls)
        if not all(c['passed'] for c in controls):
            self.stop_reason = self.stop_reason or 'control_gate_failed'
            self.halt.set()
            return self.report()
        canary = []
        for cohort in ('egolifeqa', 'week', 'day', 'month'):
            group = [r for r in self.inputs['rows'] if r['cohort'] == cohort and r['messages'] is not None]
            canary.extend(group[round(i * (len(group) - 1) / 4)] for i in range(5))
        self.parallel(canary)
        costs = {}
        for r in canary:
            valid = next((a for a in self.records[r['id']] if a['state'] == 'valid'), None)
            if valid is None or 'cost_usd' not in valid:
                self.stop_reason = self.stop_reason or 'canary_incomplete'
                self.halt.set()
                return self.report()
            costs.setdefault(r['cohort'], []).append(valid['cost_usd'])
        attempts = [a for r in canary for a in self.records[r['id']] if a['state'] != 'not_sampled']
        factor = max(1.2, len(attempts) / len(canary))
        projected = sum(max(costs[c]) * sum(r['cohort'] == c and r['messages'] is not None
                                          and self.result(r) is None for r in self.inputs['rows']) for c in costs) * factor
        gate = {'at_unix': time.time(), 'cohort_costs_usd': costs, 'retry_factor': factor,
                'observed_format_failure_rate': sum(a['state'] == 'format_or_identity_failure' for a in attempts) / len(attempts),
                'projected_remaining_usd': projected, 'already_spent_or_reserved_usd': self.cost(),
                'passed': projected + self.cost() <= self.policy['spend_ceiling_usd'] and not self.halt.is_set()}
        save(self.root / 'canary.json', gate)
        if gate['passed']:
            self.parallel(self.inputs['rows'])
        else:
            self.stop_reason = self.stop_reason or 'canary_spend_gate'
            self.halt.set()
        result = self.report()
        result['original_sources'] = self.verify_sources()
        result['raw_reparse'] = self.verify_raw()
        if result['all_completed']:
            save(self.root / 'result.json', result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--inputs-sha', required=True)
    args = parser.parse_args()
    with (args.root / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner = Runner(args.root, args.inputs_sha)
        runner.key = Path(runner.policy['credential']).read_text().strip()
        if not runner.key:
            raise ValueError('empty credential')
        def stop(signum, frame):
            runner.stop_reason = 'signal_' + str(signum)
            runner.halt.set()
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        result = runner.run()
        print(json.dumps({k: result[k] for k in ('all_completed', 'table', 'stop_reason', 'estimated_cost_usd_conservative_input_rate')}), flush=True)


if __name__ == '__main__':
    main()
