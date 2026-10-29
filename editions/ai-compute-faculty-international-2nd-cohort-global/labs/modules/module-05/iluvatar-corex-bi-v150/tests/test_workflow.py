"""Deterministic timing and evidence checks, independent of GPU performance."""
import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from loadgen.runner import run_benchmark
from tools.evidence import file_sha256, source_fingerprint, sut_signature
from tools.plot_roofline import diagnose

ROOT = Path(__file__).resolve().parents[1]

def load_step(name):
    spec = importlib.util.spec_from_file_location('step_' + name, ROOT / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

summary_step = load_step('04_summary')
profile_step = load_step('03_profiling')

class Clock:
    def __init__(self): self.now = 0.0
    def perf_counter(self): return self.now
    def sleep(self, seconds): self.now += seconds

class SUT:
    def __init__(self, clock): self.clock = clock
    def run_once(self): self.clock.now += .010
    def synchronize(self): pass
    def flops(self): return 2000
    def config(self): return {}

class TimingTests(unittest.TestCase):
    def test_overload_response_sla_and_repeats(self):
        clock = Clock()
        with patch('loadgen.runner.time.perf_counter', clock.perf_counter), patch('loadgen.runner.time.sleep', clock.sleep), patch('loadgen.runner._software_info', return_value={}):
            result = run_benchmark(SUT(clock), 'Server', 3, warmup=0, repeats=2,
                                   target_qps=200, arrival='uniform', sla={'p99_latency_ms': 12})
        self.assertEqual(len(result['queries']), 6)
        self.assertEqual(len(result['repetitions']), 2)
        self.assertAlmostEqual(result['latency_ms']['median'], 10)
        self.assertAlmostEqual(result['queue_latency_ms']['median'], 5)
        self.assertAlmostEqual(result['response_latency_ms']['median'], 15)
        self.assertFalse(result['sla']['passed'])
        self.assertEqual(result['sla']['timing_basis'], 'response_latency_ms')
        self.assertEqual(result['queries'][3]['queue_latency_ms'], 0)
        self.assertAlmostEqual(result['throughput']['samples_per_s'], 100)

    def test_invalid_measurements_rejected(self):
        for params in ({'repeats': 0}, {'num_queries': 0}, {'sync': False}):
            args = {'num_queries': 3, **params}
            with self.assertRaises(ValueError):
                run_benchmark(SUT(Clock()), 'Offline', **args)

    def test_duration_workload_bounds_gpu_queue(self):
        clock = Clock()
        class QueuedSUT:
            def __init__(self): self.pending = 0
            def config(self): return cfg
            def run_once(self):
                self.pending += 1
                if self.pending > 3: raise AssertionError('unbounded GPU queue')
            def synchronize(self):
                clock.now += self.pending * .01
                self.pending = 0
            def validate(self): return {'status': 'PASS'}
        cfg = {'shape': {'M': 64, 'N': 64, 'K': 64}, 'dtype': 'float16',
               'tile': {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32},
               'num_warps': 4, 'num_stages': 2}
        with patch.object(profile_step, 'build_sut', return_value=(QueuedSUT(), 'baseline')), patch.object(profile_step.time, 'perf_counter', clock.perf_counter):
            result = profile_step.run_sut_workload({'sut': cfg, 'resolved': 'autotune'}, 20, .025)
        self.assertEqual(result['iters'], 3)
        self.assertAlmostEqual(result['elapsed_s'], .03)
        self.assertEqual(result['implementation'], 'autotune')

    def test_reference_roof_does_not_diagnose_hardware(self):
        result = diagnose(341, 6, 192, 1228.8)
        self.assertEqual(result['model_region'], 'compute-side')
        self.assertEqual(result['actual_bottleneck'], 'undetermined')
        self.assertNotIn('Bottleneck: compute-bound', result['conclusion'])
        with self.assertRaises(ValueError): diagnose(1, float('nan'), 192, 1228.8)

class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name)
        self.make_fixture()
    def tearDown(self): self.tmp.cleanup()
    def write(self, name, data):
        (self.folder / (name + '.json')).write_text(json.dumps(data))
    def read(self, name): return json.loads((self.folder / (name + '.json')).read_text())
    def make_fixture(self):
        stat = {'count': 3, 'median': 1., 'p99': 1.}
        items = []
        for impl in ('torch', 'baseline', 'autotune'):
            cfg = {'backend': 'torch.matmul' if impl == 'torch' else 'triton', 'device': 'cuda:0',
                   'dtype': 'float16', 'shape': {'M': 64, 'N': 64, 'K': 64},
                   'tile': None if impl == 'torch' else {'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32},
                   'num_warps': 4, 'num_stages': 2, 'validation': {'status': 'PASS'},
                   'candidate_validation': [{'status': 'PASS'}]}
            items.append({'resolved': impl, 'sut': cfg, **{k: copy.deepcopy(stat) for k in
                          ('service_latency_ms', 'queue_latency_ms', 'response_latency_ms')},
                          'throughput': {'effective_tflops': 1., 'samples_per_s': 1000.},
                          'sla': {'enabled': True, 'passed': True, 'timing_basis': 'response_latency_ms'}})
        bench = {'schema_version': 2, 'benchmark_id': 'offline', 'source_fingerprint': source_fingerprint(ROOT),
                 'scenario': 'Offline', 'triton_available': True,
                 'workload': {'M': 64, 'N': 64, 'K': 64, 'dtype': 'float16', 'num_queries': 3, 'repeats': 1, 'total_queries': 3},
                 'comparison': items, 'runs': [{'queries': [{'service_latency_ms': 1., 'queue_latency_ms': 0., 'response_latency_ms': 1.}] * 3,
                                              'repetitions': [{}]} for _ in items]}
        self.write('benchmark_offline', bench)
        server = copy.deepcopy(bench); server.update(scenario='Server', benchmark_id='server')
        self.write('benchmark_server', server)
        self.write('env_check', {'cuda_available': True, 'checks': [{'status': 'ok'}]})
        roof = {'benchmark_id': 'offline', 'benchmark_sha256': file_sha256(self.folder / 'benchmark_offline.json'),
                'analysis_source_sha256': file_sha256(ROOT / '02_roofline.py') + file_sha256(ROOT / 'tools/plot_roofline.py'),
                'hardware': {'peaks_verified': False, 'source_json': str(ROOT / 'configs/hardware_bi_v150.json'),
                             'hardware_sha256': file_sha256(ROOT / 'configs/hardware_bi_v150.json')},
                'diagnoses': [{**diagnose(341, 6, 192, 1228.8), 'name': i['resolved'],
                               'sut_signature': sut_signature(i['sut'])} for i in items]}
        self.write('roofline', roof)
        tools = []
        for name in ('ixsmi', 'ixsys', 'ixkn-cli'):
            path = self.folder / (name + '.artifact'); path.write_bytes(b'fixture-capture')
            tools.append({'tool': name, 'captured': True, 'returncode': 0,
                          'artifact': {'path': str(path), 'sha256': file_sha256(path)}})
        profiling = {'capture_requested': True, 'capture_source_sha256': file_sha256(ROOT / '03_profiling.py'),
                     'benchmark_sha256': roof['benchmark_sha256'], 'benchmark_id': 'offline',
                     'workload': {'implementation': 'baseline', 'sut': items[1]['sut'], 'validation': {'status': 'PASS'}},
                     'iluvatar_tools': tools}
        self.write('diagnosis', {'profiling': profiling, 'preliminary_answers': {'implementation': 'baseline'},
                                'roofline': {'matched_diagnosis': roof['diagnoses'][1]}})
    def check(self): return summary_step.validate_results(self.folder)
    def test_complete_fixture_passes(self): self.assertEqual(self.check()['status'], 'PASS')
    def test_missing_files_fail(self):
        (self.folder / 'benchmark_server.json').unlink()
        self.assertEqual(self.check()['status'], 'FAIL')
    def test_corrupt_json_fails(self):
        (self.folder / 'benchmark_server.json').write_text('{')
        self.assertEqual(self.check()['status'], 'FAIL')
    def test_changed_benchmark_invalidates_dependents(self):
        d = self.read('benchmark_offline'); d['benchmark_id'] = 'new'; self.write('benchmark_offline', d)
        self.assertEqual(self.check()['status'], 'FAIL')
    def test_discovery_is_not_capture(self):
        d = self.read('diagnosis'); d['profiling']['capture_requested'] = False; self.write('diagnosis', d)
        self.assertEqual(self.check()['status'], 'FAIL')
    def test_missing_artifact_fails(self):
        (self.folder / 'ixsys.artifact').unlink()
        self.assertEqual(self.check()['status'], 'FAIL')
    def test_ixkn_crash_warns_but_passes(self):
        d = self.read('diagnosis')
        d['profiling']['iluvatar_tools'][-1] = {
            'tool': 'ixkn-cli', 'available': True, 'captured': False,
            'returncode': -11, 'error': 'Capture failed or produced no nonempty artifact; see log'}
        self.write('diagnosis', d)
        result = self.check()
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['errors'], [])
        self.assertTrue(any('Optional ixkn evidence unavailable' in w and 'returncode=-11' in w
                            for w in result['warnings']))
    def test_missing_ixkn_warns_but_passes(self):
        d = self.read('diagnosis'); d['profiling']['iluvatar_tools'].pop()
        self.write('diagnosis', d)
        result = self.check()
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(any('Optional ixkn evidence unavailable' in w for w in result['warnings']))
    def test_unavailable_ixkn_warns_but_passes(self):
        d = self.read('diagnosis')
        d['profiling']['iluvatar_tools'][-1] = {
            'tool': 'ixkn-cli', 'available': False, 'captured': False,
            'error': 'ixkn-cli not found in PATH'}
        self.write('diagnosis', d)
        result = self.check()
        self.assertEqual(result['status'], 'PASS')
        self.assertTrue(any('Optional ixkn evidence unavailable' in w for w in result['warnings']))
    def test_missing_or_changed_ixkn_artifact_warns_but_passes(self):
        path = self.folder / 'ixkn-cli.artifact'
        for content in (None, b'changed-capture'):
            with self.subTest(content=content):
                if content is None: path.unlink()
                else: path.write_bytes(content)
                result = self.check()
                self.assertEqual(result['status'], 'PASS')
                self.assertTrue(any('ixkn-cli: artifact missing or changed' in w for w in result['warnings']))
    def test_required_captures_still_fail_with_optional_ixkn_crash(self):
        for name in ('ixsmi', 'ixsys'):
            with self.subTest(tool=name):
                d = self.read('diagnosis')
                for tool in d['profiling']['iluvatar_tools']:
                    if tool['tool'] in (name, 'ixkn-cli'):
                        tool.update(captured=False, returncode=-11)
                    else:
                        tool.update(captured=True, returncode=0)
                self.write('diagnosis', d)
                result = self.check()
                self.assertEqual(result['status'], 'FAIL')
                self.assertTrue(any(name + ': capture missing or failed' in e for e in result['errors']))
    def test_changed_required_artifact_still_fails(self):
        (self.folder / 'ixsmi.artifact').write_bytes(b'changed-capture')
        result = self.check()
        self.assertEqual(result['status'], 'FAIL')
        self.assertTrue(any('ixsmi: artifact missing or changed' in e for e in result['errors']))
    def test_wrong_diagnosis_is_not_accepted(self):
        d = self.read('diagnosis'); d['roofline']['matched_diagnosis'] = self.read('roofline')['diagnoses'][0]
        self.write('diagnosis', d)
        self.assertEqual(self.check()['status'], 'FAIL')
    def test_failed_environment_fails(self):
        self.write('env_check', {'cuda_available': True, 'checks': [{'status': 'fail'}]})
        self.assertEqual(self.check()['status'], 'FAIL')
    def test_required_impl_missing_fails(self):
        d = self.read('benchmark_server'); d['comparison'].pop(); self.write('benchmark_server', d)
        self.assertEqual(self.check()['status'], 'FAIL')
    def test_failed_response_sla_fails(self):
        d = self.read('benchmark_server'); d['comparison'][0]['sla']['passed'] = False
        self.write('benchmark_server', d)
        self.assertEqual(self.check()['status'], 'FAIL')
    def test_service_based_server_sla_fails(self):
        d = self.read('benchmark_server'); d['comparison'][0]['sla']['timing_basis'] = 'service_latency_ms'
        self.write('benchmark_server', d)
        self.assertEqual(self.check()['status'], 'FAIL')
    def test_changed_measurement_code_fails(self):
        d = self.read('benchmark_server'); d['source_fingerprint'] = 'obsolete'
        self.write('benchmark_server', d)
        self.assertEqual(self.check()['status'], 'FAIL')
    def test_profiler_rejects_missing_implementation(self):
        d = self.read('benchmark_offline'); d['comparison'] = d['comparison'][:1]; self.write('benchmark_offline', d)
        with self.assertRaises(ValueError): profile_step.select_benchmark(self.folder / 'benchmark_offline.json', 'baseline')
    def test_profiler_rejects_different_roofline(self):
        d = self.read('diagnosis')['profiling']; d['benchmark_sha256'] = 'wrong'
        with self.assertRaises(ValueError): profile_step.merge_diagnosis(self.folder / 'roofline.json', d, self.folder / 'bad.json')

if __name__ == '__main__': unittest.main()
