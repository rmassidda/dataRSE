"""JMLR experiment package for abstract-concrete causal graph comparison."""

import os

# Prevent BLAS/OpenMP libraries (used transitively by numpy/scipy) from
# spawning a thread pool per worker process. Without this, each process
# started by ProcessPoolExecutor(max_workers=n_workers) will itself try to
# use *all* available cores for linear algebra ops, causing massive
# oversubscription (e.g. n_workers=64 on a 128-core machine still saturates
# all 128 cores). Parallelism here is achieved at the process level via
# `n_workers`, so each process should stick to a single thread.
#
# This must run before numpy/scipy/pandas are imported anywhere in the
# package, so it lives here in `__init__.py` (executed on first import of
# the `jmlr_exp` package, regardless of which submodule triggers it).
_THREAD_ENV_VARS = [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]
for _var in _THREAD_ENV_VARS:
    os.environ.setdefault(_var, "1")
