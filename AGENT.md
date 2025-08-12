# AGENT.md

**Purpose:** This document tells any human or AI assistant *how to work on* `Lsst-Twilight-SNe` safely and productively. It encodes our philosophy, process, and project‑specific guardrails (LSST/astropy/SNANA/SIMLIB).

---

## Project Facts (read first)

- **Primary goal:** Plan and simulate LSST twilight observations for SNe with outputs that are **SIMLIB‑friendly** and easy to ingest in SNANA workflows.
- **Key packages & standards:** `astropy`, `numpy`, `pandas`, `sncosmo`, LSST/OpSim/SNANA conventions; type hints; `black` + `isort`.
- **Source layout:**
  - `twilight_planner_pkg/` — library code (keep functions small & testable)
  - `notebook/` — demonstration & analysis notebooks (don’t put core logic here)
  - `data/` — small reference data only (no large or licensed data in repo)
- **Outputs:** Prefer tabular products that map cleanly to SIMLIB fields (e.g., MJD, BAND, ZP, SKYSIG, PSF, EXPTIME, RA, DEC, etc.).
- **Numerical hygiene:** Units must be explicit (e.g., degrees vs radians, seconds vs minutes); document assumptions and cite sources for LSST hardware/ops numbers.

---

## Development Guidelines

### Philosophy

- **Incremental progress over big bangs** — Small changes that compile and pass tests.
- **Learning from existing code** — Study patterns in `twilight_planner_pkg/` before coding.
- **Pragmatic over dogmatic** — Adapt to project realities (twilight constraints, SIMLIB needs).
- **Clear intent over clever code** — Be boring and obvious.

**Simplicity means:**
- Single responsibility per function/module.
- Avoid premature abstractions.
- Prefer straightforward data transforms; document any heuristics (e.g., m5 estimates).

---

## Process

### 1) Planning & Staging

For any feature > ~50 LOC or touching scheduling logic, create or update `IMPLEMENTATION_PLAN.md`:

```markdown
## Stage N: [Name]
**Goal**: [Specific deliverable]
**Success Criteria**: [Testable outcomes & sample CLI/notebook usage]
**Tests**: [Exact pytest cases + any golden files]
**Status**: [Not Started|In Progress|Complete]
```

- Keep 3–5 stages.
- Update status as you progress.
- Remove the file when all stages ship (the PR should link to it).

### 2) Implementation Flow

1. **Understand** — Read similar functions in `twilight_planner_pkg/` and existing notebooks.
2. **Test** — Write a failing unit test first (red).
3. **Implement** — Minimal code to pass (green).
4. **Refactor** — Clarify names, split long functions, keep types.
5. **Commit** — With a message that explains the *why* and links your plan or issue.

### 3) When Stuck (max 3 attempts → stop)

After **three** failed attempts on an issue, STOP and document:

- What you tried (commands/snippets), exact errors, and your hypothesis.
- 2–3 alternative approaches (with pointers to code/papers/repos).
- Re‑question fundamentals: is the abstraction right? Can we split it? Is there a simpler path?
- Propose a different angle (different API; remove an abstraction; produce a simpler output then iterate).

Open a draft PR or an issue with this note and ask for review.

---

## Technical Standards

### Architecture

- **Composition over inheritance**; pass dependencies explicitly (e.g., site/location objects, config dicts).
- **Interfaces over singletons**; accept configuration via function args or dataclasses.
- **Explicit data flow**; return plain dataclasses or typed `DataFrame`s with documented columns.
- **Test-driven when possible**; never disable tests—fix them.

### Code Quality

Every commit must:
- **Build & type‑check:** `pytest -q`, `mypy twilight_planner_pkg`, `ruff .`, `black --check .`, `isort --check-only .`
- **Pass tests**, including new tests for new behavior.
- **Follow formatting/linting**; if you must reformat, do it in a separate “format only” commit.

Commit messages: first line ≤ 72 chars, explain **why**, reference issue/plan.

### Error Handling

- Fail fast with **descriptive** exceptions (include input parameters & units).
- Handle errors **at the right layer**; library functions raise, CLI/notebooks catch & explain.
- Never silently swallow exceptions. Log warnings with actionable next steps.

---

## Decision Framework

When multiple valid approaches exist, choose by:

1. **Testability** — Can we test this without network or external services?
2. **Readability** — Will it be clear to a new contributor in 6 months?
3. **Consistency** — Match patterns already in `twilight_planner_pkg/`.
4. **Simplicity** — Smallest solution that works.
5. **Reversibility** — Easy to roll back or change.

---

## Project Integration (LSST/SNANA specifics)

### SIMLIB‑friendly outputs

- Prefer writer functions like `to_simlib_rows(df)` that emit the canonical columns:
  - `MJD, BAND, EXPTIME, ZP, ZPERR, SKYSIG, PSF1, PSF2, PSFRAT, RA, DEC, GAIN, RDNOISE`
- Keep **units and derivations** alongside code comments (e.g., how ZP/ZPERR computed; m5→ZP mapping).
- Provide a **round‑trip test**: write → read (via SNANA utilities or a local parser) → compare key fields.

### Photometric depth / m5 heuristics

- Any m5 estimator must:
  - State assumptions (sun altitude, moon phase/sep, airmass, seeing).
  - Be **conservative** by default; add a config knob to tighten/loosen.
  - Be regression‑tested with fixed inputs → fixed outputs.
- If you adjust depth → SNR → exposure time, document the chain.

### Twilight scheduling & constraints

- All altitude/azimuth computations must specify:
  - **Location** (LSST latitude/longitude, elevation),
  - **Time scale** (UTC vs TAI—use `astropy.time.Time` explicitly),
  - **Frame** (`AltAz` with pressure/temp if relevant),
  - **Sun/Moon** source and ephemerides versions.
- Add tests that pin known nights (small golden CSVs) to detect drift after refactors.

### Data policy

- Do **not** commit large binaries or restricted data.
- If needed, add a small **toy fixture** (≤ ~100 KB) for tests and point to external paths via env vars.

---

## Quality Gates

### Definition of Done (checklist)

- [ ] Tests added & passing locally and in CI
- [ ] No `ruff`/`black`/`isort`/`mypy` complaints
- [ ] Docstrings updated; public functions have examples
- [ ] Changelog entry (if user‑visible)
- [ ] Implementation matches `IMPLEMENTATION_PLAN.md`
- [ ] No stray TODOs without issue links

### Test Guidelines

- Test **behavior**, not implementation details.
- One logical assertion per test when possible; name tests as scenarios.
- Use stable seeds (`np.random.default_rng(123)`) for deterministic tests.
- Provide at least one **SIMLIB round‑trip** or schema validation test for any new output.

---

## How to Use Me (LLM Agent Operating Rules)

When asked to change code or propose a refactor:

1. **Read before writing:** skim `twilight_planner_pkg/`, find 2–3 nearest neighbors, mirror their style.
2. **Propose minimal diffs** first; show a patch that compiles and includes tests.
3. **Be explicit about units** and references (e.g., LSST tech note IDs if used).
4. **Guard numerics:** if a heuristic is introduced (e.g., m5), add:
   - an inline derivation comment,
   - a config flag,
   - 2–3 table‑driven tests.
5. **Stop after 3 failed tries**; open a note with the failure log + alternatives.

### PR Template (use this in the PR body)

- **Goal:** …  
- **Plan Stage:** … (link to `IMPLEMENTATION_PLAN.md`)  
- **Changes:** short bullet list  
- **Assumptions & Units:** explicit list  
- **Tests:** commands & what they verify  
- **SIMLIB Impact:** fields added/changed; round‑trip status  
- **Risk & Rollback:** how to revert; what could break

---

## Tooling & Commands

- Format & lint:  
  `ruff . && black . && isort .`
- Type check:  
  `mypy twilight_planner_pkg`
- Tests:  
  `pytest -q`
- Pre‑commit (recommended):  
  `pre-commit install` (run formatters/linters on commit)

---

## Important Reminders

**NEVER**
- Use `--no-verify` to bypass hooks.
- Disable tests; fix or mark with a linked issue and reasoned skip.
- Commit code that doesn’t compile or fails CI.
- Guess units—always state them.

**ALWAYS**
- Commit working code in small slices.
- Update the plan as you go.
- Mirror existing patterns and utilities.
- Stop after 3 failed attempts and reassess with notes.
---

## Quick Start for New Contributors

1. Fork, create a branch, and run:  
   `pip install -e .[dev] && pre-commit install`
2. Run tests & type checks:  
   `pytest -q && mypy twilight_planner_pkg`
3. Make a tiny change + a test, commit, open PR with the template above.

---

## License & Authorship

- Respect repository license. Keep copyright headers.
- If adding substantial code or data transformations from external sources, cite them in docstrings and PR.
