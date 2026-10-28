"""Measure opt-in local CUDA scaling; skip when any GPU is already occupied.

Run from a script with an installed CUDA-enabled magna, or PYTHONPATH=core.
Checks are observational, not a reservation: follow your server's allocation policy.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time


def _query(arguments):
    return subprocess.run(['nvidia-smi', *arguments], capture_output=True, text=True,
                          check=True, timeout=15).stdout.strip()


def check_idle():
    """Reject compute contexts even at 0% utilization, or unexplained device use."""
    processes = _query(['--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                        '--format=csv,noheader,nounits'])
    devices = _query(['--query-gpu=index,uuid,name,utilization.gpu,memory.used',
                      '--format=csv,noheader,nounits'])
    if processes:
        raise RuntimeError('GPU compute processes are present:\n'+processes)
    for line in devices.splitlines():
        fields = [x.strip() for x in line.split(',')]
        if len(fields) != 5 or int(fields[-2]) != 0 or int(fields[-1]) > 16:
            raise RuntimeError('GPU is busy or its idle status is uncertain:\n'+line)
    if not devices:
        raise RuntimeError('no GPU inventory returned')
    return devices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True, help='example directory containing run.json')
    parser.add_argument('--devices', type=int, nargs='+', required=True, help='explicit logical CUDA IDs')
    parser.add_argument('--counts', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--shots', type=int, default=16_777_216)
    parser.add_argument('--batch-size', type=int, default=65536)
    parser.add_argument('--repeat', type=int, default=3)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--proposal-scale', help='override physical sampling with this importance scale')
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1 or args.shots < 1 or not args.counts or any(n < 1 or n > len(args.devices) for n in args.counts):
        parser.error('positive shots/repeat required; every count must fit the explicit device list')
    if len(set(args.devices)) != len(args.devices) or any(d < 0 for d in args.devices):
        parser.error('devices must be distinct nonnegative indices')
    report = dict(status='started', timestamp=datetime.now(timezone.utc).isoformat(),
                  platform=platform.platform(), python=platform.python_version(),
                  arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                  runs=[])
    try:
        report['source_commit'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
            text=True, stderr=subprocess.DEVNULL, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        report['source_commit'] = None
    try:
        report['gpu_inventory'] = check_idle()
        time.sleep(2)
        check_idle()
    except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
        report.update(status='skipped', reason=str(exc))
        args.output.write_text(json.dumps(report, indent=2)+'\n')
        print('Skipped: '+str(exc))
        return 2
    # Import optional CUDA dependencies only after the occupancy check.
    from magna import compile_circuit
    from magna.cuda_multi import MultiGpuCudaSampler
    import cupy as cp
    import numpy as np
    config = json.loads((args.task/'run.json').read_text())
    sampling = (dict(method='importance', proposal_scale=args.proposal_scale)
                if args.proposal_scale else config.get('sampling'))
    compiled = compile_circuit(args.task/config['circuit'],
                               task=args.task/config['task'], options=config.get('options'), sampling=sampling)
    report.update(cupy=cp.__version__, numpy=np.__version__)
    reference = {}
    try:
        for count in args.counts:
            check_idle()
            with MultiGpuCudaSampler(compiled, devices=args.devices[:count], timeout=args.timeout) as sampler:
                cold = sampler.run(args.batch_size*count, seed=args.seed, batch_size=args.batch_size)
                rounds = []
                for repeat in range(args.repeat):
                    result = sampler.run(args.shots, seed=args.seed+repeat, batch_size=args.batch_size)
                    # Same draw IDs on every configuration; only reductions differ.
                    if repeat in reference:
                        for section in ('estimates', 'standard_errors'):
                            for key, expected in reference[repeat][section].items():
                                actual = result[section][key]
                                if expected is None:
                                    if actual is not None:
                                        raise AssertionError(f'{section}.{key}: inconsistent target presence')
                                else:
                                    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-15)
                    else:
                        reference[repeat] = result
                    rounds.append(result)
            elapsed = statistics.median(r['timings']['total_seconds'] for r in rounds)
            report['runs'].append(dict(devices=args.devices[:count], cold=cold, rounds=rounds,
                                       median_seconds=elapsed, shots_per_second=args.shots/elapsed))
            # Release contexts before checking occupancy for the next configuration.
            time.sleep(1)
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
