"""Fork-only batch-size experiment with separate memory probes.

Production CUDA kernels and native limits are unchanged. --experimental-limit
explicitly overrides the Python sampler's metadata for larger device batches;
it never calls a native CPU bridge with oversized input. Each worker owns its
GPU context. The parent never initializes CUDA. One- and eight-worker timings
include the same process-dispatch and aggregation path.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
from time import perf_counter
import traceback

helper_path = Path(__file__).with_name('run-eight-gpu-20261028.py')
spec = importlib.util.spec_from_file_location('eight_gpu_benchmark', helper_path)
helper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helper)
PROBE_SEED = 987654321


def measured_worker(connection, device, snapshot):
    """Keep the production worker protocol; write probe-only telemetry to disk."""
    try:
        import cupy as cp
        from magna.cuda_multi import _compile_snapshot
        from magna.cuda_sampler import CudaSampler, _merge_moments
        settings = snapshot['_benchmark']
        with cp.cuda.Device(device):
            sampler = CudaSampler(_compile_snapshot(snapshot))
            original_limit = sampler._experiment.metadata['max_shots']
            limit = settings['limit']
            if limit > original_limit and not settings['experimental']:
                raise ValueError('oversized batch requires explicit experimental override')
            sampler._experiment.metadata['max_shots'] = limit
            pool = cp.get_default_memory_pool()
            pool.set_limit(size=settings['pool_limit_bytes'])
            features = [m['feature_count'] for m in sampler._queries.metadata]
            # Kernels use uint64 for shots and linear array indices. Enforce
            # CUDA grid-x and Python/int64 allocation-size bounds separately.
            if max([limit, *(limit*f for f in features)]) > (2**31-1)*128:
                raise ValueError('experimental batch would exceed conservative grid bound')
            connection.send(('ready', None))
            while True:
                command = connection.recv()
                if command is None:
                    return
                shots, seed, offset, batch = command
                sampler._experiment.validate()
                probe = seed == PROBE_SEED
                if probe:
                    cp.cuda.get_current_stream().synchronize()
                    gc.collect()
                    pool.free_all_blocks()
                    free, total = cp.cuda.runtime.memGetInfo()
                    memory = dict(device=device, batch_size=batch, shots=shots,
                                  total_bytes=total, baseline_device_used=total-free,
                                  peak_device_used=total-free,
                                  peak_live_pool_bytes=pool.used_bytes(),
                                  peak_reserved_pool_bytes=pool.total_bytes(),
                                  frame_words=sampler._experiment.width,
                                  query_count=sampler._queries.query_count,
                                  feature_counts=features, native_limit=original_limit,
                                  effective_limit=limit)

                    def allocate(size):
                        pointer = pool.malloc(size)
                        free, total = cp.cuda.runtime.memGetInfo()
                        memory['peak_device_used'] = max(memory['peak_device_used'], total-free)
                        memory['peak_live_pool_bytes'] = max(memory['peak_live_pool_bytes'], pool.used_bytes())
                        memory['peak_reserved_pool_bytes'] = max(memory['peak_reserved_pool_bytes'], pool.total_bytes())
                        return pointer

                    cp.cuda.set_allocator(allocate)
                try:
                    started = perf_counter()
                    moments = _merge_moments(sampler._chunks(shots, seed, offset, batch))
                    elapsed = perf_counter()-started
                finally:
                    if probe:
                        cp.cuda.set_allocator(pool.malloc)
                if probe:
                    cp.cuda.get_current_stream().synchronize()
                    free, total = cp.cuda.runtime.memGetInfo()
                    memory['peak_device_used'] = max(memory['peak_device_used'], total-free)
                    memory['probe_seconds'] = elapsed
                    path = Path(settings['memory_dir'])/f'device-{device}-batch-{batch}.json'
                    path.write_text(json.dumps(memory, indent=2)+'\n')
                connection.send(('result', (moments, elapsed)))
    except BaseException:
        try:
            connection.send(('error', traceback.format_exc()))
        except (OSError, EOFError):
            pass
    finally:
        connection.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--reference-results', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--batches', nargs='+', type=int, default=[65536, 131072, 262144, 524288, 1000000])
    p.add_argument('--counts', nargs='+', type=int, default=[8])
    p.add_argument('--tasks', nargs='+')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--shot-multiplier', type=int, default=4)
    p.add_argument('--shots', type=int)
    p.add_argument('--experimental-limit', type=int, default=1000000)
    p.add_argument('--pool-limit-gib', type=float, default=18.)
    args = p.parse_args()
    if (args.repeat < 1 or args.shot_multiplier < 1 or not args.batches or
            min(args.batches) < 1 or max(args.batches) > args.experimental_limit or
            not set(args.counts) <= {1, 8} or not 0 < args.pool_limit_gib <= 18):
        p.error('invalid bounds/counts; the device allocation pool is capped at 18 GiB')
    uuids = tuple(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    if len(uuids) != 8 or len(set(uuids)) != 8 or not all(u.startswith('GPU-') for u in uuids):
        p.error('exactly eight distinct GPU UUIDs required')
    helper.process_guard(uuids)
    reference_bytes = args.reference_results.read_bytes()
    reference = json.loads(reference_bytes)
    if reference['status'] != 'complete':
        p.error('reference must be complete')
    memory_root = args.output.resolve().with_suffix('.memory')
    memory_root.mkdir(exist_ok=False)
    report = dict(status='started', source_commit=args.source_commit,
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                  timestamp=datetime.now(timezone.utc).isoformat(),
                  python=platform.python_version(), platform=platform.platform(),
                  selected_gpu_uuids=uuids, initial_inventory=helper.inventory(),
                  initial_load=os.getloadavg(), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  batches=args.batches, counts=args.counts, repeats=args.repeat,
                  experimental_limit=args.experimental_limit,
                  pool_limit_bytes=int(args.pool_limit_gib*2**30), tasks=[])

    def save():
        pending = args.output.with_suffix('.tmp')
        pending.write_text(json.dumps(report, indent=2)+'\n')
        pending.replace(args.output)

    save()
    try:
        import numpy as np
        from magna import compile_circuit
        from magna import cuda_multi
        report['numpy'] = np.__version__
        cuda_multi._worker_main = measured_worker
        for previous in reference['tasks']:
            name = previous['task']
            if args.tasks and name not in args.tasks:
                continue
            root = Path('examples')/name
            config = json.loads((root/'run.json').read_text())
            compiled = compile_circuit(root/config['circuit'], task=root/config['task'],
                                       options=config.get('options'), sampling=previous['sampling'])
            shots = args.shots or max(previous['shots']*args.shot_multiplier, 8*max(args.batches))
            item = dict(task=name, sampling=previous['sampling'], shots=shots, configurations=[])
            report['tasks'].append(item)
            for count in args.counts:
                helper.process_guard(uuids)
                memory_dir = memory_root/f'{name}-{count}'
                memory_dir.mkdir()
                with cuda_multi.MultiGpuCudaSampler(compiled, devices=range(count), timeout=1200) as sampler:
                    sampler._validation._experiment.metadata['max_shots'] = args.experimental_limit
                    sampler._snapshot['_benchmark'] = dict(
                        limit=args.experimental_limit, experimental=args.experimental_limit>1000000,
                        pool_limit_bytes=report['pool_limit_bytes'], memory_dir=str(memory_dir))
                    sampler.run(count*65536, seed=7, batch_size=65536)
                    owners = {w.process.pid for w in sampler._workers.values()}
                    configurations = {}
                    # Probe two full batches per worker, with monitoring entirely
                    # excluded from the subsequent ordinary timed sampling calls.
                    for batch in args.batches:
                        helper.process_guard(uuids, owners)
                        sampler.run(2*count*batch, seed=PROBE_SEED, batch_size=batch)
                        memory = [json.loads((memory_dir/f'device-{d}-batch-{batch}.json').read_text())
                                  for d in range(count)]
                        configurations[batch] = dict(count=count, batch_size=batch, memory=memory, rounds=[])
                        item['configurations'].append(configurations[batch])
                        print(f'{name} {count} GPUs batch={batch}: memory probe '
                              f"max={max(m['peak_device_used'] for m in memory)/2**30:.3f} GiB", flush=True)
                        save()
                    for repeat in range(args.repeat):
                        shift = repeat % len(args.batches)
                        order = args.batches[shift:]+args.batches[:shift]
                        baseline = None
                        for batch in order:
                            helper.process_guard(uuids, owners)
                            # Ensure the exact allocation shape is cached. No
                            # allocator monitoring is active in this warmup/run.
                            sampler.run(count*batch, seed=13, batch_size=batch)
                            tick = perf_counter()
                            result = sampler.run(shots, seed=20261028+repeat, batch_size=batch)
                            wall = perf_counter()-tick
                            helper.process_guard(uuids, owners)
                            if result['status'] != 'complete' or result['completed'] != shots:
                                raise AssertionError('incomplete run')
                            if baseline is None:
                                baseline = result
                            differences = helper.check_results(baseline, result, np)
                            row = dict(seed=20261028+repeat, order=order, wall_seconds=wall,
                                       result=result, absolute_differences=differences, load=os.getloadavg())
                            configurations[batch]['rounds'].append(row)
                            save()
                            print(f'{name} {count} GPUs batch={batch} round={repeat+1}: '
                                  f'{wall:.6f}s {shots/wall/1e6:.3f} million/s', flush=True)
                    for entry in configurations.values():
                        times = [r['wall_seconds'] for r in entry['rounds']]
                        entry.update(median_seconds=statistics.median(times), min_seconds=min(times),
                                     max_seconds=max(times), shots_per_second=shots/statistics.median(times))
                save()
            del compiled
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
