#!/usr/bin/env python3
"""Profile the exact Triton configuration recorded by an Offline benchmark."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from kernels.gemm_triton import build_sut
from tools.evidence import file_sha256, source_fingerprint, sut_signature


def select_benchmark(path: Path, implementation: str) -> tuple[dict, dict]:
    bench = json.loads(path.read_text())
    if bench.get('schema_version') != 2 or bench.get('scenario') != 'Offline':
        raise ValueError('Use a schema-v2 Offline benchmark; rerun 01_benchmark.py')
    if bench.get('source_fingerprint') != source_fingerprint(ROOT):
        raise ValueError('Benchmark was produced by different code; rerun 01_benchmark.py')
    matches = [x for x in bench.get('comparison', []) if x.get('resolved') == implementation]
    if len(matches) != 1 or matches[0]['sut'].get('validation', {}).get('status') != 'PASS':
        raise ValueError(f'No unique validated {implementation} in {path}')
    return bench, matches[0]


def run_sut_workload(item: dict, iters: int, duration: float = 0.0) -> dict:
    cfg = item['sut']
    shape = cfg['shape']
    tile = {**cfg['tile'], 'num_warps': cfg['num_warps'], 'num_stages': cfg['num_stages']}
    # Reuse the selected launch even for autotune: do not retune under a profiler.
    sut, resolved = build_sut('baseline', shape['M'], shape['N'], shape['K'], cfg['dtype'], tile)
    if resolved != 'baseline' or sut_signature(sut.config()) != sut_signature(cfg):
        raise ValueError('Runtime SUT does not match the selected GPU configuration')
    for _ in range(3):
        sut.run_once()
    sut.synchronize()
    start = time.perf_counter()
    count = 0
    while ((time.perf_counter() - start < duration) if duration else (count < iters)):
        sut.run_once()
        if duration:
            # Bound GPU queue growth so a 5-second sampler load drains promptly.
            sut.synchronize()
        count += 1
    sut.synchronize()
    elapsed = time.perf_counter() - start
    validation = sut.validate()
    return {'implementation': item['resolved'], 'launch_mode': 'selected-fixed-configuration',
            'warmup': 3, 'iters': count, 'elapsed_s': elapsed, 'sut': sut.config(),
            'validation': validation}


def run_capture(tool: str, command: list[str], output: Path, log: Path) -> dict:
    info = {'tool': tool, 'available': bool(shutil.which(tool)), 'captured': False,
            'command': command, 'log': str(log)}
    if not info['available']:
        info['error'] = f'{tool} not found in PATH'
        return info
    # Never accept an artifact left by an earlier failed capture.
    output.unlink(missing_ok=True)
    try:
        with log.open('w') as stream:
            proc = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=180)
        info['returncode'] = proc.returncode
        info['captured'] = proc.returncode == 0 and output.is_file() and output.stat().st_size > 0
        if info['captured']:
            info['artifact'] = {'path': str(output), 'sha256': file_sha256(output), 'bytes': output.stat().st_size}
        else:
            info['error'] = 'Capture failed or produced no nonempty artifact; see log'
    except (OSError, subprocess.TimeoutExpired) as exc:
        info['error'] = str(exc)
    return info


def merge_diagnosis(roofline_path: Path, profiling: dict, out_path: Path) -> dict:
    roof = json.loads(roofline_path.read_text())
    if roof.get('benchmark_sha256') != profiling['benchmark_sha256']:
        raise ValueError('Roofline and profiling refer to different benchmark contents; rerun 02_roofline.py')
    impl = profiling['workload']['implementation']
    matches = [d for d in roof.get('diagnoses', []) if d.get('name') == impl]
    if len(matches) != 1 or matches[0].get('sut_signature') != sut_signature(profiling['workload']['sut']):
        raise ValueError('Roofline item does not match the profiled implementation/configuration')
    item = matches[0]
    result = {'schema_version': 2, 'experiment': '03_diagnosis', 'profiling': profiling,
              'roofline': {'benchmark_sha256': roof['benchmark_sha256'], 'matched_diagnosis': item},
              'preliminary_answers': {'implementation': impl, 'model_region': item['model_region'],
                                      'actual_bottleneck': 'undetermined', 'from_roofline': item['conclusion']}}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, default=ROOT / 'results/benchmark_offline.json')
    parser.add_argument('--impl', choices=('baseline', 'autotune'), default='baseline')
    parser.add_argument('--iters', type=int, default=20)
    parser.add_argument('--duration', type=float, default=0.0, help='sustained workload seconds, overrides iters')
    for name in ('M', 'N', 'K'):
        parser.add_argument('--' + name, type=int, default=None, help='must match the benchmark')
    parser.add_argument('--dtype', default=None, help='must match the benchmark')
    parser.add_argument('--workload-only', action='store_true')
    parser.add_argument('--capture', action='store_true', help='actually collect dmon, ixsys and ixkn reports')
    parser.add_argument('--enable-proton', action='store_true', help='unsupported on BI-V150')
    parser.add_argument('--roofline', type=Path, default=ROOT / 'results/roofline.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'results/diagnosis.json')
    args = parser.parse_args()
    if args.iters <= 0 or not math.isfinite(args.duration) or args.duration < 0:
        parser.error('iters must be positive; duration must be finite and nonnegative')
    if args.enable_proton:
        parser.error('Proton is unsupported on BI-V150; use --capture')
    if not args.workload_only:
        args.out.unlink(missing_ok=True)
        (args.out.parent / "profiling/profiling_summary.json").unlink(missing_ok=True)
    bench, item = select_benchmark(args.benchmark, args.impl)
    for name in ('M', 'N', 'K'):
        if getattr(args, name) is not None and getattr(args, name) != item['sut']['shape'][name]:
            parser.error('Shape override differs from benchmark; rerun benchmark first')
    if args.dtype is not None and args.dtype != item['sut']['dtype']:
        parser.error('dtype override differs from benchmark')
    workload = run_sut_workload(item, args.iters, args.duration)
    print(f"[workload] {args.impl}: {workload['iters']} launches, {workload['elapsed_s']:.4f}s", flush=True)
    if args.workload_only:
        return 0
    out_dir = args.out.parent / 'profiling'
    out_dir.mkdir(parents=True, exist_ok=True)
    child = [sys.executable, str(ROOT / '03_profiling.py'), '--benchmark', str(args.benchmark.resolve()),
             '--impl', args.impl, '--workload-only']
    tools = [{'tool': t, 'available': bool(shutil.which(t)), 'captured': False,
              'status': 'not_requested'}
             for t in ('ixsmi', 'ixsys', 'ixkn-cli')]
    if args.capture:
        duration = args.duration or 5.0
        samples = math.ceil(duration) + 8
        sampler = out_dir / 'ixsmi_dmon.log'
        sampling = {'tool': 'ixsmi', 'available': bool(shutil.which('ixsmi')), 'captured': False}
        if sampling['available']:
            with sampler.open('w') as stream:
                proc = subprocess.Popen(['ixsmi', 'dmon', '-s', 'pucvt', '-c', str(samples)],
                                        stdout=stream, stderr=subprocess.STDOUT)
                try:
                    load = subprocess.run([*child, '--duration', str(duration)], stdout=subprocess.DEVNULL,
                                          stderr=subprocess.STDOUT, timeout=180)
                    sampling['returncode'] = proc.wait(timeout=samples + 20)
                    values = [int(v[4]) for line in sampler.read_text().splitlines()
                              if (v := line.split()) and v[0] == '0' and len(v) > 4]
                    sampling['max_util_percent'] = max(values, default=0)
                    sampling['sample_count'] = len(values)
                    sampling['captured'] = (load.returncode == 0 and proc.returncode == 0 and
                                            len(values) > 0 and max(values) > 0)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait()
            if sampling['captured']:
                sampling['artifact'] = {'path': str(sampler), 'sha256': file_sha256(sampler),
                                        'bytes': sampler.stat().st_size}
        tools = [sampling]
        trace = out_dir / 'gemm.trace'
        info = run_capture('ixsys', ['ixsys', '-t', 'cuda', '-o', str(trace), *child], trace, out_dir / 'ixsys.log')
        if info['captured']:
            try:
                with sqlite3.connect(f'file:{trace.resolve()}?mode=ro', uri=True) as db:
                    count = db.execute("SELECT count(*) FROM cuda_kernel_data k JOIN function_name_map f "
                                       "ON f.id=k.name_id WHERE f.name='_matmul_kernel'").fetchone()[0]
                info['gemm_kernel_count'] = count
                info['captured'] = count > 0
            except sqlite3.Error as exc:
                info.update(captured=False, error=f'Trace could not be validated: {exc}')
        tools.append(info)
        report = out_dir / 'gemm_kernel.ixkn-rep'
        info = run_capture('ixkn-cli', ['ixkn-cli', '--set', 'default', '--kernel-name', '_matmul_kernel',
                           '--launch-skip', '3', '--launch-count', '1', '--export-profile',
                           str(out_dir / 'gemm_kernel'), *child], report, out_dir / 'ixkn.log')
        if info['captured']:
            with (out_dir / 'ixkn_details.log').open('w') as stream:
                proc = subprocess.run(['ixkn-cli', '--import-profile', str(report), '--page', 'details'],
                                      stdout=stream, stderr=subprocess.STDOUT, timeout=60)
            info['captured'] = proc.returncode == 0 and 'Kernel: _matmul_kernel' in (out_dir / 'ixkn_details.log').read_text()
        tools.append(info)
    profiling = {'schema_version': 2, 'capture_source_sha256': file_sha256(Path(__file__)), 'benchmark_id': bench['benchmark_id'],
                 'benchmark_sha256': file_sha256(args.benchmark), 'workload': workload,
                 'capture_requested': args.capture, 'iluvatar_tools': tools,
                 'proton': {'available': False, 'captured': False, 'status': 'unsupported_on_BI-V150'}}
    diag = merge_diagnosis(args.roofline, profiling, args.out)
    (out_dir / 'profiling_summary.json').write_text(json.dumps(profiling, indent=2))
    for tool in tools:
        print(f"[{ 'CAPTURED' if tool['captured'] else 'NOT CAPTURED'}] {tool['tool']}")
    print(f"[model] {args.impl}: {diag['preliminary_answers']['model_region']}; actual bottleneck undetermined")
    return 1 if args.capture and not all(t['captured'] for t in tools) else 0


if __name__ == '__main__':
    raise SystemExit(main())
