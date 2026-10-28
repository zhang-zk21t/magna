"""Interleaved 1/4/8-GPU comparison at the previous four-GPU draw counts.

Run from the repository root with exactly eight idle GPUs selected by UUID.
The reference JSON supplies draw counts only; every speed is measured anew.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter


def command(args):
    return subprocess.check_output(args, text=True, timeout=15).strip()


def process_guard(uuids, allowed=()):
    rows = command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_gpu_memory',
                    '--format=csv,noheader,nounits'])
    for row in rows.splitlines():
        uuid, pid, memory = (s.strip() for s in row.split(','))
        if uuid in uuids and int(pid) not in allowed:
            raise RuntimeError('another process is using a selected GPU: '+row)
    return rows


def inventory():
    return command(['nvidia-smi',
                    '--query-gpu=index,uuid,name,driver_version,utilization.gpu,memory.used,temperature.gpu,power.draw,clocks.sm',
                    '--format=csv'])


def check_results(single, multi, np):
    delta = {}
    for section in ('estimates', 'standard_errors'):
        delta[section] = {}
        assert single[section].keys() == multi[section].keys()
        for name, expected in single[section].items():
            actual = multi[section][name]
            if expected is None:
                if actual is not None:
                    raise AssertionError(f'{section}.{name}: different target semantics')
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-15)
                delta[section][name] = abs(actual-expected)
    return delta


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--reference-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=65536)
    args = p.parse_args()
    uuids = tuple(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    if len(uuids) != 8 or len(set(uuids)) != 8 or not all(u.startswith('GPU-') for u in uuids):
        p.error('CUDA_VISIBLE_DEVICES must contain exactly eight distinct GPU UUIDs')
    if args.repeat < 1 or args.batch_size < 1:
        p.error('repeat and batch-size must be positive')
    reference_bytes = args.reference_results.read_bytes()
    reference = json.loads(reference_bytes)
    if reference['status'] != 'complete':
        p.error('reference measurements must be complete')
    process_guard(uuids)
    report = dict(status='started', source_commit=args.source_commit,
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                  timestamp=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), platform=platform.platform(),
                  selected_gpu_uuids=uuids, initial_inventory=inventory(),
                  initial_load=os.getloadavg(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  batch_size=args.batch_size, repeats=args.repeat, tasks=[])

    def save():
        pending = args.output.with_suffix('.tmp')
        pending.write_text(json.dumps(report, indent=2)+'\n')
        pending.replace(args.output)

    save()
    try:
        import cupy as cp
        import numpy as np
        from magna import compile_circuit
        from magna.cuda_sampler import CudaSampler
        from magna.cuda_multi import MultiGpuCudaSampler
        report.update(cupy=cp.__version__, numpy=np.__version__,
                      cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                      cuda_driver=cp.cuda.runtime.driverGetVersion())
        if cp.cuda.runtime.getDeviceCount() != 8:
            raise RuntimeError('CUDA visibility did not expose exactly eight devices')
        modes = ('single', 'four', 'eight')
        for task_index, previous in enumerate(reference['tasks']):
            process_guard(uuids, (os.getpid(),))
            name, shots, proposal = previous['task'], previous['shots'], previous['sampling']
            root = Path('examples')/name
            config = json.loads((root/'run.json').read_text())
            compiled = compile_circuit(root/config['circuit'], task=root/config['task'],
                                       options=config.get('options'), sampling=proposal)
            item = dict(task=name, shots=shots, sampling=proposal, initial_inventory=inventory(),
                        load_before=os.getloadavg(), first_calls={}, rounds=[])
            report['tasks'].append(item)
            with cp.cuda.Device(0), ExitStack() as stack:
                samplers = {'single': CudaSampler(compiled)}
                for mode, count in (('four', 4), ('eight', 8)):
                    samplers[mode] = stack.enter_context(
                        MultiGpuCudaSampler(compiled, devices=range(count), timeout=600))
                owners = {os.getpid()}
                for mode in modes:
                    process_guard(uuids, owners)
                    tick = perf_counter()
                    result = samplers[mode].run(args.batch_size*8, seed=7, batch_size=args.batch_size)
                    item['first_calls'][mode] = dict(seconds=perf_counter()-tick, result=result)
                    if mode != 'single':
                        owners.update(w.process.pid for w in samplers[mode]._workers.values())
                    process_guard(uuids, owners)
                item['worker_pids'] = sorted(owners)
                # Equal unmeasured warmup for all three modes after pool startup.
                for mode in modes:
                    process_guard(uuids, owners)
                    samplers[mode].run(1_048_576, seed=13, batch_size=args.batch_size)
                    process_guard(uuids, owners)
                for repeat in range(args.repeat):
                    # Rotate both across rounds and tasks to vary scheduling/order effects.
                    shift = (repeat+task_index) % len(modes)
                    order = modes[shift:]+modes[:shift]
                    row = dict(seed=20261028+repeat, order=order, load=os.getloadavg())
                    for mode in order:
                        process_guard(uuids, owners)
                        tick = perf_counter()
                        result = samplers[mode].run(shots, seed=row['seed'], batch_size=args.batch_size)
                        row[mode] = dict(wall_seconds=perf_counter()-tick, result=result)
                        if result['status'] != 'complete' or result['completed'] != shots:
                            raise AssertionError('incomplete run')
                        process_guard(uuids, owners)
                    row['absolute_differences'] = {
                        mode: check_results(row['single']['result'], row[mode]['result'], np)
                        for mode in ('four', 'eight')}
                    row['inventory_after'] = inventory()
                    item['rounds'].append(row)
                    save()
                    timings = ' '.join(f"{mode}={row[mode]['wall_seconds']:.6f}s" for mode in modes)
                    print(f'{name} round {repeat+1}: {timings}', flush=True)
                for mode in modes:
                    times = [r[mode]['wall_seconds'] for r in item['rounds']]
                    median = statistics.median(times)
                    item[mode] = dict(median_seconds=median, min_seconds=min(times), max_seconds=max(times),
                                      shots_per_second=shots/median)
                item['four_over_single'] = item['single']['median_seconds']/item['four']['median_seconds']
                item['eight_over_single'] = item['single']['median_seconds']/item['eight']['median_seconds']
                item['eight_over_four'] = item['four']['median_seconds']/item['eight']['median_seconds']
                item['load_after'] = os.getloadavg()
            del compiled, samplers
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            save()
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['final_inventory'] = inventory()
        save()


if __name__ == '__main__':
    main()
