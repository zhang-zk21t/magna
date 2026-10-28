"""Paired direct-single-GPU versus persistent-four-GPU sampling experiment.

Set CUDA_VISIBLE_DEVICES to exactly four GPU UUIDs before Python starts.
Only these GPUs are checked/used; unrelated contexts on other GPUs are preserved.
Source revision is supplied because the remote experiment uses a git archive.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
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
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=65536)
    p.add_argument('--target-seconds', type=float, default=2.)
    p.add_argument('--shots', type=int, help='fixed shots instead of pilot-selected workload')
    p.add_argument('--tasks', nargs='+', default=['bh2-T', 'bh100-T', 'color3', 'rp2-d5'])
    args = p.parse_args()
    uuids = tuple(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    if len(uuids) != 4 or len(set(uuids)) != 4 or not all(u.startswith('GPU-') for u in uuids):
        p.error('CUDA_VISIBLE_DEVICES must contain exactly four distinct GPU UUIDs')
    process_guard(uuids)
    report = dict(status='started', source_commit=args.source_commit,
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
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
        if cp.cuda.runtime.getDeviceCount() != 4:
            raise RuntimeError('CUDA visibility did not expose exactly four devices')
        scales = {'bh2-T': '1000/7', 'bh100-T': '500/77'}
        for name in args.tasks:
            process_guard(uuids, (os.getpid(),))
            root = Path('examples')/name
            config = json.loads((root/'run.json').read_text())
            proposal = (dict(method='importance', proposal_scale=scales[name])
                        if name in scales else dict(method='physical'))
            compiled = compile_circuit(root/config['circuit'], task=root/config['task'],
                                       options=config.get('options'), sampling=proposal)
            item = dict(task=name, sampling=proposal, initial_inventory=inventory(),
                        load_before=os.getloadavg(), rounds=[])
            report['tasks'].append(item)
            with cp.cuda.Device(0), MultiGpuCudaSampler(compiled, devices=range(4), timeout=600) as multi:
                single = CudaSampler(compiled)
                tick = perf_counter()
                single.run(args.batch_size*4, seed=7, batch_size=args.batch_size)
                item['single_first_call_seconds'] = perf_counter()-tick
                tick = perf_counter()
                multi.run(args.batch_size*4, seed=7, batch_size=args.batch_size)
                item['four_first_call_seconds'] = perf_counter()-tick
                owners = {os.getpid(), *(w.process.pid for w in multi._workers.values())}
                process_guard(uuids, owners)
                pilot_shots = max(1_048_576, args.batch_size*4)
                tick = perf_counter()
                pilot = single.run(pilot_shots, seed=13, batch_size=args.batch_size)
                pilot_time = perf_counter()-tick
                quantum = args.batch_size*4
                selected = min(134_217_728, max(pilot_shots,
                    math.ceil(args.target_seconds/pilot_time*pilot_shots/quantum)*quantum))
                shots = args.shots if args.shots is not None else selected
                item.update(shots=shots, pilot_shots=pilot_shots, pilot_seconds=pilot_time,
                            pilot_result=pilot, worker_pids=sorted(owners))
                # Prime the same workload shape on four devices before measured rounds.
                multi.run(pilot_shots, seed=13, batch_size=args.batch_size)
                for repeat in range(args.repeat):
                    order = ('single', 'four') if repeat % 2 == 0 else ('four', 'single')
                    row = dict(seed=20261028+repeat, order=order, load=os.getloadavg())
                    for mode in order:
                        process_guard(uuids, owners)
                        sampler = single if mode == 'single' else multi
                        tick = perf_counter()
                        result = sampler.run(shots, seed=row['seed'], batch_size=args.batch_size)
                        row[mode] = dict(wall_seconds=perf_counter()-tick, result=result)
                        process_guard(uuids, owners)
                    row['absolute_differences'] = check_results(row['single']['result'], row['four']['result'], np)
                    row['speedup'] = row['single']['wall_seconds']/row['four']['wall_seconds']
                    row['inventory_after'] = inventory()
                    item['rounds'].append(row)
                    save()
                    print(f"{name} round {repeat+1}: single={row['single']['wall_seconds']:.6f}s "
                          f"four={row['four']['wall_seconds']:.6f}s speedup={row['speedup']:.3f}x", flush=True)
                for mode in ('single', 'four'):
                    times = [r[mode]['wall_seconds'] for r in item['rounds']]
                    median = statistics.median(times)
                    item[mode] = dict(median_seconds=median, min_seconds=min(times), max_seconds=max(times),
                                      shots_per_second=shots/median)
                item['speedup_ratio_of_medians'] = item['single']['median_seconds']/item['four']['median_seconds']
                item['parallel_efficiency'] = item['speedup_ratio_of_medians']/4
                item['load_after'] = os.getloadavg()
                del single
            del compiled, multi
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
