#!/usr/bin/env python3
"""Validate experiment provenance and completeness, then summarize the results."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.evidence import file_sha256, source_fingerprint, sut_signature


def validate_results(results: Path, required: tuple[str, ...] = ('torch', 'baseline', 'autotune')) -> dict:
    errors, warnings, tables = [], [], {}
    data = {}
    for name in ('env_check', 'benchmark_offline', 'benchmark_server', 'roofline', 'diagnosis'):
        path = results / (name + '.json')
        try:
            value = json.loads(path.read_text())
            if not isinstance(value, dict) or not value:
                raise ValueError('expected a nonempty JSON object')
            data[name] = value
        except (OSError, ValueError) as exc:
            errors.append(f'{name}: {exc}')
    env = data.get('env_check', {})
    checks = env.get('checks', [])
    if not isinstance(checks, list) or any(not isinstance(c, dict) for c in checks):
        checks = []
    if not env.get('cuda_available') or not checks or any(c.get('status') == 'fail' for c in checks):
        errors.append('Environment must report CUDA available and no failed checks')
    if any(c.get('status') == 'warn' for c in checks):
        warnings.append('Environment contains warnings; inspect env_check.json')
    impls = {}
    for scenario in ('offline', 'server'):
        name = 'benchmark_' + scenario
        d = data.get(name)
        if not d:
            continue
        try:
            if d.get('schema_version') != 2 or d.get('source_fingerprint') != source_fingerprint(ROOT):
                raise ValueError('legacy result or different measurement code; rerun benchmark')
            if d['scenario'].lower() != scenario or d['triton_available'] is not True:
                raise ValueError('wrong scenario or no Triton runtime')
            items = d['comparison']
            mapped = {x['resolved']: x for x in items}
            if len(mapped) != len(items) or not set(required) <= set(mapped):
                raise ValueError('required implementations missing or duplicated: ' + ', '.join(required))
            total = d['workload']['num_queries'] * d['workload']['repeats']
            if total <= 0 or d['workload']['total_queries'] != total:
                raise ValueError('invalid query/repeat counts')
            runs = d['runs']
            if len(runs) != len(items):
                raise ValueError('run/comparison count mismatch')
            for item in items:
                cfg = item['sut']
                if 'fallback' in item['resolved'] or not cfg['device'].startswith('cuda'):
                    raise ValueError('CPU/fallback result is not a GPU lab result')
                if cfg['shape'] != {k: d['workload'][k] for k in ('M', 'N', 'K')} or cfg['dtype'] != d['workload']['dtype']:
                    raise ValueError('SUT shape/dtype differs from benchmark workload')
                if cfg['backend'] == 'triton':
                    if cfg.get('validation', {}).get('status') != 'PASS':
                        raise ValueError('Triton numerical validation missing or failed')
                    if item['resolved'] == 'autotune' and (not cfg.get('candidate_validation') or
                            any(c.get('status') != 'PASS' for c in cfg['candidate_validation'])):
                        raise ValueError('autotune candidate validation missing or failed')
                for key in ('service_latency_ms', 'queue_latency_ms', 'response_latency_ms'):
                    stat = item[key]
                    if stat['count'] != total or not all(math.isfinite(stat[k]) and stat[k] >= 0 for k in ('median', 'p99')):
                        raise ValueError('invalid latency sample counts or values')
                if scenario == 'server' and item['sla']['timing_basis'] != 'response_latency_ms':
                    raise ValueError('Server SLA must use response latency')
                if item['sla'].get('enabled') and not item['sla'].get('passed'):
                    raise ValueError('configured SLA failed')
                if not math.isfinite(item['throughput']['effective_tflops']) or item['throughput']['effective_tflops'] <= 0:
                    raise ValueError('invalid effective throughput')
            for run in runs:
                if len(run['queries']) != total or len(run['repetitions']) != d['workload']['repeats']:
                    raise ValueError('measured queries/repetitions do not match declared counts')
                for q in run['queries']:
                    if not math.isclose(q['response_latency_ms'], q['service_latency_ms'] + q['queue_latency_ms'], abs_tol=1e-6):
                        raise ValueError('response != service + queue')
            impls[scenario] = mapped
            tables[scenario] = [{
                'implementation': i['resolved'], 'service_median_ms': i['service_latency_ms']['median'],
                'response_p99_ms': i['response_latency_ms']['p99'],
                'samples_per_s': i['throughput']['samples_per_s'],
                'effective_tflops': i['throughput']['effective_tflops'], 'samples': total,
            } for i in items]
            if total < 100:
                warnings.append(f'{scenario}: {total} samples; p99 is exploratory')
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f'{name}: {exc}')
    if len(impls) == 2:
        a, b = data['benchmark_offline']['workload'], data['benchmark_server']['workload']
        if any(a[k] != b[k] for k in ('M', 'N', 'K', 'dtype')):
            errors.append('Offline and Server shapes/dtypes differ')
        if 'baseline' in impls['offline'] and 'baseline' in impls['server']:
            if sut_signature(impls['offline']['baseline']['sut']) != sut_signature(impls['server']['baseline']['sut']):
                errors.append('Offline and Server baseline configurations differ')
    roof = data.get('roofline', {})
    diag = data.get('diagnosis', {})
    offline = data.get('benchmark_offline', {})
    try:
        digest = file_sha256(results / 'benchmark_offline.json')
        if roof.get('benchmark_sha256') != digest or roof.get('benchmark_id') != offline.get('benchmark_id'):
            raise ValueError('Roofline is stale or belongs to a different benchmark')
        if roof.get('analysis_source_sha256') != file_sha256(ROOT / '02_roofline.py') + file_sha256(ROOT / 'tools/plot_roofline.py'):
            raise ValueError('Roofline analysis code changed; rerun 02_roofline.py')
        hw = roof['hardware']
        if file_sha256(Path(hw['source_json'])) != hw['hardware_sha256']:
            raise ValueError('Hardware reference JSON changed since Roofline')
        if hw.get('student_todos_sha256') and file_sha256(ROOT / 'configs/student_todos.py') != hw['student_todos_sha256']:
            raise ValueError('Student peak/implementation settings changed since Roofline')
        models = {x['name']: x for x in roof['diagnoses']}
        if set(models) != set(impls.get('offline', {})):
            raise ValueError('Roofline implementations differ from Offline benchmark')
        for name, item in impls.get('offline', {}).items():
            model = models[name]
            if model['sut_signature'] != sut_signature(item['sut']):
                raise ValueError('Roofline configuration differs from benchmark')
            if model.get('actual_bottleneck') != 'undetermined':
                raise ValueError('Roofline must not claim the measured bottleneck')
        if not roof['hardware']['peaks_verified']:
            warnings.append('Roofline uses unverified reference peak assumptions')
        p = diag['profiling']
        if p['benchmark_sha256'] != digest or p['benchmark_id'] != offline.get('benchmark_id'):
            raise ValueError('Profiling is stale or belongs to a different benchmark')
        if p.get('capture_source_sha256') != file_sha256(ROOT / '03_profiling.py'):
            raise ValueError('Profiling code changed; recapture results')
        w = p['workload']
        name = w['implementation']
        if sut_signature(w['sut']) != sut_signature(impls['offline'][name]['sut']):
            raise ValueError('Profiled SUT differs from benchmark')
        if w['validation']['status'] != 'PASS':
            raise ValueError('Profiled SUT validation did not pass')
        if diag['preliminary_answers']['implementation'] != name or diag['roofline']['matched_diagnosis'] != models[name]:
            raise ValueError('Diagnosis refers to a different implementation/model')
        tools = {x['tool']: x for x in p['iluvatar_tools']}
        if not p.get('capture_requested'):
            raise ValueError('Only tool discovery was done; run 03_profiling.py --capture')
        for name in ('ixsmi', 'ixsys', 'ixkn-cli'):
            try:
                tool = tools.get(name, {})
                if not tool.get('captured') or tool.get('returncode') != 0:
                    raise ValueError(f'{name}: capture missing or failed (returncode={tool.get("returncode")})')
                artifact = tool['artifact']
                path = Path(artifact['path'])
                if not path.is_file() or path.stat().st_size == 0 or file_sha256(path) != artifact['sha256']:
                    raise ValueError(f'{name}: artifact missing or changed since capture')
            except (OSError, KeyError, TypeError, ValueError) as exc:
                if name != 'ixkn-cli':
                    raise
                warnings.append(f'Optional ixkn evidence unavailable: {exc}; kernel-level counters are not validated')
    except (OSError, KeyError, TypeError, ValueError) as exc:
        errors.append(f'Roofline/profiling consistency: {exc}')
    return {'experiment': '04_summary', 'schema_version': 2, 'status': 'FAIL' if errors else 'PASS',
            'errors': errors, 'warnings': warnings, 'tables': tables}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT / 'results')
    parser.add_argument('--required-impl', nargs='+', default=['torch', 'baseline', 'autotune'])
    args = parser.parse_args()
    result = validate_results(args.results, tuple(args.required_impl))
    args.results.mkdir(parents=True, exist_ok=True)
    (args.results / 'summary.json').write_text(json.dumps(result, indent=2))
    for scenario, rows in result['tables'].items():
        print(f'== {scenario}: service median / response p99 (ms) ==')
        for row in rows:
            print(f"{row['implementation']:<10} {row['service_median_ms']:.3f} / "
                  f"{row['response_p99_ms']:.3f}  {row['effective_tflops']:.3f} TFLOP/s  n={row['samples']}")
    for level in ('warnings', 'errors'):
        for message in result[level]:
            print(f'[{level}] {message}')
    print(f"[acceptance] {result['status']} (warnings do not certify hardware peaks or tail stability)")
    return 1 if result['errors'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
