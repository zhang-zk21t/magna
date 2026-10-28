"""Compare the public single-process CPU run API with eight GPUs at equal draws.

Both samplers reuse compiled tasks. CPU and GPU RNG algorithms differ, so
statistics are compared using standard errors, not exact seeded equality.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
from time import perf_counter, process_time

helper_path = Path(__file__).with_name('run-eight-gpu-20261028.py')
spec = importlib.util.spec_from_file_location('eight_gpu_benchmark', helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--reference-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shots', type=int, default=262144)
    p.add_argument('--repeat', type=int, default=5)
    args = p.parse_args()
    uuids = tuple(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    if len(uuids) != 8 or len(set(uuids)) != 8 or not all(u.startswith('GPU-') for u in uuids):
        p.error('exactly eight distinct GPU UUIDs required')
    if args.shots < 2 or args.repeat < 1:
        p.error('shots must be at least two and repeat must be positive')
    helper.process_guard(uuids)
    reference_bytes = args.reference_results.read_bytes()
    reference = json.loads(reference_bytes)
    if reference['status'] != 'complete':
        p.error('reference measurements must be complete')
    report = dict(status='started', source_commit=args.source_commit,
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                  timestamp=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), platform=platform.platform(),
                  selected_gpu_uuids=uuids, initial_inventory=helper.inventory(),
                  initial_load=os.getloadavg(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  thread_limits={k: os.environ.get(k) for k in
                                 ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                  shots=args.shots, batch_size=65536, repeats=args.repeat, tasks=[])

    def save():
        pending = args.output.with_suffix('.tmp')
        pending.write_text(json.dumps(report, indent=2)+'\n')
        pending.replace(args.output)

    save()
    try:
        import numpy as np
        from magna import compile_circuit, run
        from magna.cuda_multi import MultiGpuCudaSampler
        report['numpy'] = np.__version__
        for previous in reference['tasks']:
            helper.process_guard(uuids, (os.getpid(),))
            name, proposal = previous['task'], previous['sampling']
            root = Path('examples')/name
            config = json.loads((root/'run.json').read_text())
            compiled = compile_circuit(root/config['circuit'], task=root/config['task'],
                                       options=config.get('options'), sampling=proposal)
            item = dict(task=name, sampling=proposal, shots=args.shots,
                        load_before=os.getloadavg(), rounds=[])
            report['tasks'].append(item)
            with MultiGpuCudaSampler(compiled, devices=range(8), timeout=600) as multi:
                tick = perf_counter()
                warm = run(compiled, shots=4096, seed=7)
                if warm['status'] != 'complete':
                    raise RuntimeError(f'CPU warmup failed: {warm}')
                item['cpu_warmup_seconds'] = perf_counter()-tick
                tick = perf_counter()
                multi.run(args.shots, seed=7, batch_size=65536)
                item['eight_first_call_seconds'] = perf_counter()-tick
                owners = {os.getpid(), *(w.process.pid for w in multi._workers.values())}
                for repeat in range(args.repeat):
                    order = ('cpu', 'eight') if repeat % 2 == 0 else ('eight', 'cpu')
                    row = dict(seed=20261028+repeat, order=order, load=os.getloadavg())
                    for mode in order:
                        helper.process_guard(uuids, owners)
                        tick, cpu_tick = perf_counter(), process_time()
                        result = (run(compiled, shots=args.shots, seed=row['seed']) if mode == 'cpu'
                                  else multi.run(args.shots, seed=row['seed'], batch_size=65536))
                        row[mode] = dict(wall_seconds=perf_counter()-tick,
                                         parent_cpu_seconds=process_time()-cpu_tick, result=result)
                        if result['status'] != 'complete' or result['completed'] != args.shots:
                            raise AssertionError('incomplete run')
                        helper.process_guard(uuids, owners)
                    # Different RNGs: record differences relative to combined sampling errors.
                    row['estimate_z_differences'] = {}
                    for key in ('acceptance', 'error', 'conditional_error'):
                        a, b = (row[m]['result']['estimates'][key] for m in ('cpu', 'eight'))
                        if a is None or b is None:
                            if a != b:
                                raise AssertionError('different target semantics')
                            continue
                        sa, sb = (row[m]['result']['standard_errors'][key] for m in ('cpu', 'eight'))
                        denominator = math.hypot(sa, sb)
                        if denominator == 0 and a != b:
                            raise AssertionError('different deterministic estimates')
                        row['estimate_z_differences'][key] = abs(a-b)/denominator if denominator else 0.
                    row['parent_os_threads'] = len(list(Path('/proc/self/task').iterdir()))
                    row['inventory_after'] = helper.inventory()
                    item['rounds'].append(row)
                    save()
                    print(f"{name} round {repeat+1}: cpu={row['cpu']['wall_seconds']:.6f}s "
                          f"eight={row['eight']['wall_seconds']:.6f}s "
                          f"speedup={row['cpu']['wall_seconds']/row['eight']['wall_seconds']:.2f}x", flush=True)
                for mode in ('cpu', 'eight'):
                    times = [r[mode]['wall_seconds'] for r in item['rounds']]
                    median = statistics.median(times)
                    item[mode] = dict(median_seconds=median, min_seconds=min(times), max_seconds=max(times),
                                      shots_per_second=args.shots/median)
                item['eight_over_cpu'] = item['cpu']['median_seconds']/item['eight']['median_seconds']
                item['load_after'] = os.getloadavg()
            del compiled, multi
            gc.collect()
            save()
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['finished_at'] = datetime.now(timezone.utc).isoformat()
        report['final_inventory'] = helper.inventory()
        save()


if __name__ == '__main__':
    main()
