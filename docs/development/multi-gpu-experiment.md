# Multi-GPU sampling experiment

This fork-only experiment builds on the final head of upstream PR #5,
`5a0e201b420c09625f77c59889bbc60b6ed8a478`. The branch is
`experiment/multi-gpu-sampling`. No upstream pull request is part of this experiment.

## Implementation

The [multi-GPU API](../software/cuda.md#multiple-gpus-on-one-host-experimental)
uses one persistent spawned process per explicitly selected GPU. Each process
reconstructs the compiled task from a source/configuration snapshot, selects its
device, and retains the existing single-device sampler. The CUDA kernels are unchanged.

Draw intervals are balanced, contiguous and disjoint. All shards use one seed
with distinct global offsets; changing device count preserves the sampled faults.
Each worker returns a count, two means and a three-entry centered covariance
factor. The parent merges these in draw-interval order with PR #5's Chan/QR
implementation, then calculates estimates and standard errors. Memory used for
aggregation scales with the number of workers, not the number of shots or batches.

The parent handles worker exceptions, disconnected pipes, hard process exits,
interruptions and optional wall-time deadlines. A failed run closes the pool and
raises without publishing a partial estimate. Empty shards start no processes.
The implementation supports one host, static splitting and the task subset of #5.

## Server preflight on 2026-10-28

Only read-only SSH inspection was performed on `ghz`. Contrary to the expected
eight devices, `nvidia-smi` enumerated nine RTX 4090 GPUs, indices 0 through 8.
GPU 0 reported 492 MiB used; the compute-process query reported PID 866579 using
482 MiB, even though device utilization was 0% and the default process table was
empty. The other eight devices reported 2 MiB and 0% utilization. The host's load
averages were approximately 120, 122 and 111.

The user's instruction was to abandon server experiments if another job occupied
the server. The GPU context was sufficient to skip the experiment. No environment
was installed, no task was copied and no sampling was started on `ghz`. No attempt
was made to use the remaining idle devices or terminate the existing process.

**Real multi-GPU correctness, throughput and 1/2/4/8-card scaling remain unverified.**
CPU-reference tests do not establish device isolation, CUDA runtime compatibility,
GPU memory consumption or speedup on that server.

## Local validation

Local Ubuntu under WSL2, Python 3.11.4, GCC 11.4.0, NumPy 2.1.2 and Stim 1.16.0:

- Full source regression: `Ran 413 tests`, `OK (skipped=19)`. The skip entries
  comprise six CUDA device test classes and thirteen installed-wheel tests,
  which are run separately against the built wheel.
- The new host-side checks include thirteen multiprocessing/statistics tests
  (with up to eight actual child processes) and four occupancy-guard tests.
- Documentation language/link checks and the strict MkDocs 1.6.1 build passed.
- A Linux CPython 3.11 wheel was built successfully, with no CUDA dependency
  required for the build.
- All thirteen installed-wheel acceptance tests passed in a clean environment
  with neither NumPy nor CuPy. They also verify that the new module is packaged
  without causing the ordinary CPU API to import optional dependencies.

The first environment check exposed a local Clang 9 installation missing its
UBSan runtime and a missing optional Stim dependency. Validation was repeated
after installing Stim in the isolated environment and using `/usr/bin` GCC for
the native sanitizer tests. No repository test was relaxed or skipped to handle
these environment failures. Reproduce that compiler selection with:

```sh
env PATH=/usr/bin:/bin OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 CXX=g++ \
    PYTHONPATH=core:. .venv/bin/python scripts/project_checks.py --suite current
```

## Reproduction

Host-side tests use actual spawned processes and the independent CPU references
provided by #5, including eight-worker physical/importance runs, an independent
per-draw NumPy covariance check, offsets crossing 2^32, uint64 interval bounds,
empty/small runs, target/recovery snapshots, process reuse, failure and cleanup.
Separate benchmark-guard tests reject compute contexts at zero utilization.

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=core:. \
    python -m unittest discover -s tests/core -p 'test_cuda_multi*.py' -v
```

After a separate allocation and idle check, device tests can be explicitly enabled:

```sh
MAGNA_TEST_CUDA_DEVICES=0,1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=core:. \
    python -m unittest discover -s tests/core -p test_cuda_multi.py -v
```

The [benchmark script](../../scripts/benchmark_multi_gpu.py) measures cold and
warmed runs, checks numerical agreement across device counts, records the source
commit and saves JSON. It rejects any initially occupied GPU visible to
`nvidia-smi`; it is not a scheduler or device reservation mechanism. For eight
allocated devices:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=core \
    python scripts/benchmark_multi_gpu.py --task examples/bh100-T \
    --devices 0 1 2 3 4 5 6 7 --counts 1 2 4 8 --proposal-scale 500/77 \
    --shots 16777216 --repeat 3 --output multi-gpu-results.json
```

Keep cold startup/JIT time separate from warmed throughput. The default workload
is a starting point, not a claim that it saturates every GPU. No extrapolated
performance numbers are presented here.
