"""
Internal modules for the twilight planner scheduler.

This package must be *side-effect free* at import time. In particular,
**do not import from the parent package** (e.g. ``..scheduler``) because
``twilight_planner_pkg.scheduler`` imports these submodules and Python
first executes our ``__init__``; importing back up to the parent creates
an initialization cycle and breaks test collection.

Submodules are imported directly (e.g. ``from .caching import ...``) by
``twilight_planner_pkg.scheduler``. We intentionally do not re-export
the parent’s public API here.
"""

# Keep init light and with no heavy imports or re-exports.
__all__: list[str] = []
