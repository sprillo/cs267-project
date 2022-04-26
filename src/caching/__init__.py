"""
Caching decorator module.

The caching decorator module provides decorators to wrap functions such that
any previously done computation is avoided in subsequent function calls. The
caching is performed directly in the filesystem (as opposed to in-memory),
enabling persistent caching. This is fundamental for compute-intensive
applications.
"""
from ._cached_computation import cached_computation
from ._cached_parallel_computation import cached_parallel_computation
from ._common import set_cache_dir, set_hash, set_log_level