## Stage 1: Aggregate SIMLIB epochs by supernova
**Goal**: group all scheduled epochs for a single SN into one SIMLIB LIBID block.
**Success Criteria**: scheduler output has one LIBID per SN with `NOBS` equal to epoch count; regression test spans multiple nights.
**Tests**: `pytest -q`
**Status**: Complete
