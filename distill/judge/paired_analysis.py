"""Question-paired, cohort-specific percentile bootstrap for fixed judge scores."""
import argparse
import hashlib
import json
import platform
import random
from pathlib import Path

COHORTS = ('egolifeqa', 'week', 'day', 'month')


def unique(rows):
    result = {r['id']: r for r in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate question identity')
    return result


def align(base, sft, inputs):
    expected = hashlib.sha256(json.dumps(inputs['policy'], ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    if (base.get('arm') != 'base' or base.get('base_rejudged') is not True
            or base.get('policy_sha256') != expected or sft.get('base_rejudged') is not False):
        raise ValueError('result arm or frozen policy mismatch')
    if not base['all_completed'] or not sft['all_completed'] or base['model'] != sft['model']:
        raise ValueError('incomplete or mismatched judge')
    b, s, frozen = unique(base['rows']), unique(sft['rows']), unique(inputs['rows'])
    order = inputs['pair_order']
    if len(set(order)) != len(order) or set(order) != set(b) or set(b) != set(s) or set(b) != set(frozen):
        raise ValueError('question membership mismatch')
    paired = []
    for key in order:
        if frozen[key]['question_reference_sha256'] != inputs['sft_reference_sha256'][key]:
            raise ValueError('question or reference mismatch')
        if b[key]['cohort'] != s[key]['cohort'] or b[key]['cohort'] != frozen[key]['cohort']:
            raise ValueError('cohort mismatch')
        if b[key]['score'] not in (0, .5, 1) or s[key]['score'] not in (0, .5, 1):
            raise ValueError('invalid normalized score')
        paired.append((b[key], s[key]))
    return paired


def load_results(base_path, sft_path, inputs):
    raw = Path(sft_path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != inputs['sft_provenance']['result_sha256']:
        raise ValueError('SFT result hash mismatch')
    base, sft = json.loads(Path(base_path).read_bytes()), json.loads(raw)
    align(base, sft, inputs)
    return base, sft


def bootstrap(differences, seed, replicates=100000):
    if not differences or replicates < 2:
        raise ValueError('empty cohort or too few replicates')
    rng = random.Random(seed)
    means = sorted(100*sum(rng.choices(differences, k=len(differences)))/len(differences)
                   for _ in range(replicates))
    def quantile(p):
        index = (replicates-1)*p
        lo = int(index)
        hi = min(lo+1, replicates-1)
        return means[lo]+(means[hi]-means[lo])*(index-lo)
    return [quantile(.025), quantile(.975)]


def analyze(base, sft, inputs, replicates=100000):
    paired = align(base, sft, inputs)
    rows = []
    for index, cohort in enumerate(COHORTS):
        group = [(b, s) for b, s in paired if b['cohort'] == cohort]
        if not group:
            continue
        n, seed = len(group), 20261114+index
        delta = [s['score']-b['score'] for b, s in group]
        old_delta = [s['deepseek_score']-b['deepseek_score'] for b, s in group]
        rows.append({'cohort': cohort, 'n': n, 'base_percent': 100*sum(b['score'] for b, s in group)/n,
                     'sft_percent': 100*sum(s['score'] for b, s in group)/n,
                     'delta_percentage_points': 100*sum(delta)/n,
                     'ci95_percentage_points': bootstrap(delta, seed, replicates),
                     'deepseek_delta_percentage_points': 100*sum(old_delta)/n,
                     'deepseek_ci95_percentage_points': bootstrap(old_delta, seed, replicates),
                     'bootstrap_seed': seed,
                     'base_generation_zero': sum(b['state'] == 'generation_timeout_zero' for b, s in group),
                     'sft_generation_zero': sum(s['state'] == 'generation_timeout_zero' for b, s in group)})
    return {'model': base['model'], 'both_arms_gpt5_rejudged': True, 'rows': rows,
            'bootstrap': {'replicates': replicates, 'python': platform.python_version(),
                          'rng': 'random.Random(seed).choices(differences,k=n)',
                          'order': 'frozen inputs.pair_order filtered in ego/week/day/month order',
                          'pair_order_sha256': hashlib.sha256(json.dumps(inputs['pair_order']).encode()).hexdigest(),
                          'quantile': 'linear interpolation; h=(replicates-1)*p; p=.025,.975',
                          'interval_target': 'paired SFT minus Base normalized-score mean, in percentage points',
                          'interpretation': 'question-level uncertainty conditional on fixed generations and one judge sample; not video-level or training-seed uncertainty'}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--sft', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text())
    base, sft = load_results(args.base, args.sft, inputs)
    result = analyze(base, sft, inputs)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))
