## Stage 1: Define local-night utilities
**Goal**: Add timezone helper and local-night twilight window bundling in `astro_utils`.
**Success Criteria**: Function returns evening/morning windows for a given local date, each tagged with `night_date`.
**Tests**: `pytest twilight_planner_pkg/tests/test_local_night.py`
**Status**: Complete

## Stage 2: Schedule by local night
**Goal**: Use local-night windows in `scheduler` and set discovery cutoff at 23:59:59 local.
**Success Criteria**: Scheduling operates on evening/morning pair for each civil night.
**Tests**: `pytest -q`
**Status**: Complete

## Stage 3: Final verification
**Goal**: Ensure formatting, linting, and type checks pass.
**Success Criteria**: `ruff .`, `black --check .`, `isort --check-only .`, `mypy twilight_planner_pkg`, `pytest -q` all succeed.
**Tests**: command list above
**Status**: Complete
