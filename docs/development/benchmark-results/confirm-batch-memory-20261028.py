"""Independent interleaved 1/8-device confirmation of batch sweep candidates.

One persistent eight-worker pool is reused. Between completed calls only, the
benchmark selects either its first worker or all eight by changing _devices.
This private harness avoids overlapping CUDA contexts/buffers on a device; it
does not add a production API or change the sampler implementation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
from time import perf_counter

worker_path = Path(__file__).with_name('run-batch-memory-20261028.py')
spec = importlib.util.spec_from_file_location('batch_memory_benchmark', worker_path)
batch_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(batch_module)
helper = batch_module.helper
CANDIDATES = {'bh2-T': [65536, 524288], 'bh100-T': [65536, 131072],
              'color3': [65536, 1000000, 50331648], 'rp2-d5': [65536, 524288]}


def worker(*args):
    return batch_module.measured_worker(*args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--reference-results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    uuids = tuple(os.environ.get('CUDA_VISIBLE_DEVICES', '').split(','))
    if len(uuids) != 8 or len(set(uuids)) != 8 or not all(u.startswith('GPU-') for u in uuids):
        parser.error('exactly eight GPU UUIDs required')
    helper.process_guard(uuids)
    reference_bytes = args.reference_results.read_bytes()
    reference = json.loads(reference_bytes)
    assert reference['status'] == 'complete'
    report = dict(status='started', source_commit=args.source_commit,
                  driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  worker_sha256=hashlib.sha256(worker_path.read_bytes()).hexdigest(),
                  helper_sha256=hashlib.sha256(batch_module.helper_path.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                  timestamp=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  platform=platform.platform(), initial_load=os.getloadavg(),
                  selected_gpu_uuids=uuids, initial_inventory=helper.inventory(),
                  repeats=5, seed_base=20261128, candidates=CANDIDATES, tasks=[])

    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.output)

    save()
    try:
        import numpy as np
        from magna import compile_circuit, cuda_multi
        cuda_multi._worker_main = worker
        for task_index, previous in enumerate(reference['tasks']):
            helper.process_guard(uuids)
            name = previous['task']
            batches = CANDIDATES[name]
            root = Path('examples')/name
            config = json.loads((root/'run.json').read_text())
            compiled = compile_circuit(root/config['circuit'], task=root/config['task'],
                                       options=config.get('options'), sampling=previous['sampling'])
            shots = previous['shots']*4
            cases = [(count, batch) for batch in batches for count in (1, 8)]
            entries = {(count, batch): dict(count=count, batch_size=batch, rounds=[])
                       for count, batch in cases}
            item = dict(task=name, shots=shots, sampling=previous['sampling'],
                        configurations=list(entries.values()))
            report['tasks'].append(item)
            with cuda_multi.MultiGpuCudaSampler(compiled, devices=range(8), timeout=1200) as pool:
                limit = max(1000000, max(batches))
                pool._validation._experiment.metadata['max_shots'] = limit
                pool._snapshot['_benchmark'] = dict(limit=limit, experimental=limit>1000000,
                                                     pool_limit_bytes=18*2**30, memory_dir='unused')
                pool.run(8*65536, seed=7, batch_size=65536)
                owners = {w.process.pid for w in pool._workers.values()}
                for repeat in range(5):
                    shift = (repeat+task_index) % len(cases)
                    order = cases[shift:]+cases[:shift]
                    expected = None
                    for count, batch in order:
                        helper.process_guard(uuids, owners)
                        # No run is active while the private active-device view changes.
                        pool._devices = tuple(range(count))
                        pool.run(2*count*batch, seed=13, batch_size=batch)
                        tick = perf_counter()
                        result = pool.run(shots, seed=20261128+repeat, batch_size=batch)
                        wall = perf_counter()-tick
                        helper.process_guard(uuids, owners)
                        assert result['completed'] == shots and len(result['shards']) == count
                        if expected is None:
                            expected = result
                        row = dict(seed=20261128+repeat, order=order, wall_seconds=wall,
                                   result=result, load=os.getloadavg(),
                                   absolute_differences=helper.check_results(expected, result, np))
                        entries[(count, batch)]['rounds'].append(row)
                        save()
                        print(f'{name} {count} GPUs batch={batch} round={repeat+1}: '
                              f'{wall:.6f}s {shots/wall/1e6:.3f} million/s', flush=True)
                for entry in entries.values():
                    times = [r['wall_seconds'] for r in entry['rounds']]
                    entry.update(median_seconds=statistics.median(times), min_seconds=min(times),
                                 max_seconds=max(times), shots_per_second=shots/statistics.median(times))
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
