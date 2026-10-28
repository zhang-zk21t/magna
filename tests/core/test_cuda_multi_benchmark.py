"""The benchmark must skip occupied hardware before importing CuPy."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.benchmark_multi_gpu import check_idle, main


class MultiGpuBenchmarkGuardTests(unittest.TestCase):
    def test_zero_utilization_with_compute_context_is_busy(self):
        with patch('scripts.benchmark_multi_gpu._query', side_effect=[
                'GPU-test, 866579, 482', '0, GPU-test, RTX 4090, 0, 492']):
            with self.assertRaisesRegex(RuntimeError, 'compute processes'):
                check_idle()

    def test_unexplained_memory_activity_or_missing_inventory_rejected(self):
        for inventory in ('0, GPU-test, RTX 4090, 0, 492',
                          '0, GPU-test, RTX 4090, 1, 2', '',
                          '0, GPU-test, RTX 4090, N/A, 2'):
            with self.subTest(inventory=inventory):
                with patch('scripts.benchmark_multi_gpu._query', side_effect=['', inventory]):
                    with self.assertRaises((RuntimeError, ValueError)):
                        check_idle()

    def test_idle_devices_accepted(self):
        inventory = '0, GPU-a, RTX 4090, 0, 2\n1, GPU-b, RTX 4090, 0, 2'
        with patch('scripts.benchmark_multi_gpu._query', side_effect=['', inventory]):
            self.assertEqual(check_idle(), inventory)

    def test_busy_host_writes_skipped_report_before_loading_task(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'report.json'
            argv = ['benchmark_multi_gpu.py', '--task', 'nonexistent-task',
                    '--devices', '0', '--counts', '1', '--output', str(output)]
            with patch('sys.argv', argv), patch('scripts.benchmark_multi_gpu.check_idle',
                                              side_effect=RuntimeError('occupied')):
                self.assertEqual(main(), 2)
            report = json.loads(output.read_text())
            self.assertEqual(report['status'], 'skipped')
            self.assertEqual(report['runs'], [])


if __name__ == '__main__':
    unittest.main()
