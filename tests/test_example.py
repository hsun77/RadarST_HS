"""Reproduce the included RadarST example and check its outputs."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class ExampleReproducibilityTest(unittest.TestCase):
    def test_example_outputs(self):
        repository = Path(__file__).resolve().parents[1]
        expected = json.loads((Path(__file__).parent / 'expected_example.json').read_text(encoding='utf-8'))
        environment = os.environ.copy()
        environment['PYTHONDONTWRITEBYTECODE'] = '1'
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            environment[name] = '1'
        with tempfile.TemporaryDirectory(prefix='radarst_example_', dir=repository.parent) as temporary:
            output = Path(temporary) / 'output'
            command = [sys.executable, '-B', str(repository / 'run_radarst.py'),
                       '--input-dir', str(repository / 'example_data/map'),
                       '--molecular-dir', str(repository / 'example_data/molecular'),
                       '--output-root', str(output),
                       '--dataset', expected['dataset'], '--workflow', expected['workflow']]
            completed = subprocess.run(command, cwd=repository, env=environment,
                                       capture_output=True, text=True, encoding='utf-8', timeout=600)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            prediction = output / 'predictions' / expected['dataset'] / expected['workflow'] / 'E2_EVENT_CREDIT'
            for name, checksum in expected['sha256'].items():
                with self.subTest(file=name):
                    path = prediction / name
                    self.assertTrue(path.is_file(), str(path))
                    self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)


if __name__ == '__main__':
    unittest.main()
