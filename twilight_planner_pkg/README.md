# LSST Twilight Planner

Modular planner for scheduling **Vera C. Rubin Observatory (LSST)** twilight observations of supernovae (SNe).
Optimized for fast, shallow snaps in bright sky, with options to export **SNANA SIMLIBs** for downstream simulations.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Quickstart (CLI)

```bash
python -m twilight_planner_pkg.main --csv your.csv --out results \
    --start 2024-01-01 --end 2024-01-07 \
    --lat -30.2446 --lon -70.7494 --height 2663 \
    --filters g,r,i,z --exp g:5,r:5,i:5,z:5 \
    --min_alt 20 --evening_cap 600 --morning_cap 600 \
    --per_sn_cap 120 --max_sn 10 --strategy hybrid \
    --hybrid-detections 2 --hybrid-exposure 300 \
    --simlib-out results/night.SIMLIB --simlib-survey LSST_TWILIGHT
```

To force LSST-only light curves for all SNe:

```bash
python -m twilight_planner_pkg.main --csv your.csv --out results \
    --start 2024-01-01 --end 2024-01-07 \
    --lat -30.2446 --lon -70.7494 --height 2663 \
    --strategy lc --lc-detections 5 --lc-exposure 300
```

---

## Priority Modes

1. **Discovery‑optimized** — maximize breadth with minimal repeats
2. **Hybrid (default)** — quick color (≥2 detections in ≥2 filters or ≥300 s total).
   - If Type Ia: escalate to the light‑curve goal
   - Else: drop priority (reallocate time)
3. **LSST‑only light curves** — pursue a full LC for every SN (≥5 detections or ≥300 s across ≥2 filters)

---

## Notebook Example

```python
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps

cfg = PlannerConfig(lat_deg=-30.2446, lon_deg=-70.7494, height_m=2663)
plan_twilight_range_with_caps('/path/to/your.csv', '/tmp/out',
                              '2024-01-01', '2024-01-07', cfg)
```

`PlannerConfig` highlights:
- `priority_strategy`: "hybrid" (default) or "lc"
- `hybrid_detections` / `hybrid_exposure_s`: quick‑color thresholds (defaults: ≥2 detections across ≥2 filters or ≥300 s total)
- `lc_detections` / `lc_exposure_s`: LC thresholds (defaults: ≥5 detections or ≥300 s across ≥2 filters)
- Default escalation: once hybrid is met, cached `SN_type_raw` from the CSV decides whether to escalate (Ia) or deprioritize (non‑Ia)

---

## How It Works (Algorithm)

### Inputs & Pre‑processing
- **Inputs** — candidates CSV, output directory, UTC start/end dates, and a `PlannerConfig` (site, filters, caps, overheads)
- **Columns** — RA, Dec, discovery date (ISO or MJD), name, and type are auto‑detected if not specified
- **Derived** — `RA_deg`, `Dec_deg`, `discovery_datetime` (UTC), `Name`, `SN_type_raw` are normalized/added

### Eligibility
- A target must rise above `min_alt_deg` (default 20°) at some sampled time
- Supernovae without a valid `discovery_datetime` are skipped
- Visibility window lasts `ceil(1.2 × days)` after discovery where `days`
  comes from `typical_days_by_type` or a default fallback

Key selection criteria recap:

- A supernova must have a valid discovery date within the configured observation window.
- Must be observable at altitude ≥ `min_alt_deg` (default 20°) during at least one twilight window (Sun altitude between −18° and 0°).
- Eligibility duration after discovery is scaled from a `typical_days_by_type` value (or a default if type unknown) and multiplied by 1.2 to provide a buffer.
- Observation times must also pass the Moon separation rule, which applies a filter-dependent minimum separation, waived if the Moon is below the horizon.

### Twilight Windows & Best Time
- Twilight windows are spans with Sun altitude $h_\odot \in [-18^\circ,0^\circ)$
- Sample each window every `twilight_step_min` minutes
- For each SN, choose the time of maximum altitude that passes altitude and Moon constraints (below)

### Selection & Scheduling
- Rank visible SNe by need score (how far from meeting the current goal) and altitude
- Keep the top `max_sn_per_night` globally; split by window (0=morning, 1=evening)
- Within each window, schedule via greedy nearest‑neighbor on great‑circle distance
- Enforce window caps (`morning_cap_s`, `evening_cap_s`) and per‑SN cap (`per_sn_cap_s`); trim filters greedily to fit

Prioritization strategy recap:

- Hybrid strategy (default): Aim for quick color (≥ 2 detections in ≥ 2 filters or ≥ 300 s total). Once met:
  - Type Ia SNe escalate to the LSST-only light-curve goal (≥ 5 detections or ≥ 300 s across ≥ 2 filters).
  - Non-Ia SNe drop to zero priority, freeing time for other targets.
- LSST-only light curve strategy: Every SN is pursued until the LC goal is met.
- Candidates are ranked nightly first by need score (how far from meeting the active goal) and then by their maximum altitude within the twilight window.
- Scheduling within each window uses a greedy nearest-neighbor approach on sky position to minimize slew time, constrained by per-SN and per-window time caps.

### Slews & Overheads
- Two‑regime slew: small moves ≤3.5° take ≈4 s; larger moves add $t_{\rm slew} \approx 4\,\mathrm{s} + \frac{\max(0,\Delta\theta-3.5^\circ)}{5.25^\circ/\mathrm{s}} + 1\,\mathrm{s}_{\rm settle}$
- Readout: 2 s per exposure; filter change: 120 s per swap
- Per‑SN budget: $T_{\rm SN} = t_{\rm slew} + \sum_{f \in \mathcal{F}}(t_{\rm exp}(f) + t_{\rm read}) + N_{\rm changes}\,t_{\rm filt}$

### Filters & Exposure Assignment
- Filters are requested in priority order and truncated if adding another
  would exceed `per_sn_cap_s`
- Planning continues even if requested filters exceed
  `carousel_capacity`, but a warning is issued

---

## Part 3: Science Formalism Implemented

This section documents the math implemented by the planner and used when writing SIMLIBs.
### 1) Geometry, Airmass, and Altitude

Given site latitude $\phi$, target declination $\delta$, and hour angle $H$,

the altitude $h$ and zenith distance $z = 90^\circ - h$ are

```math

\sin h = \sin\phi\,\sin\delta + \cos\phi\,\cos\delta\,\cos H.

```

The airmass (X) uses the Kasten–Young (1989) approximation (robust near twilight):

```math

X(h) = \left[\cos z + 0.50572 \left(96.07995^\circ - z\right)^{-1.6364}\right]^{-1}.

```

Eligibility requires $h \ge h_{\min}$ (default $20^\circ$) at some sampled time in a twilight window.

### 2) Moon Separation (Graded Policy)

Let $\Delta\theta_{\rm Moon}$ be the angular separation from the Moon, $f\in[0,1]$ the Moon illuminated fraction, and $h_{\rm Moon}$ the Moon altitude. The planner uses a graded minimum separation:

```math

\Delta\theta_{\min}(f,h_{\rm Moon}) := \Delta\theta_0 \Big[ 1 - \alpha f \Big] \Big[ 1 - \beta \max(0, \sin h_{\rm Moon}) \Big],

```

with band‑dependent $\Delta\theta_0$ (e.g., $g:30^\circ$, $r:25^\circ$, $i:20^\circ$, $z:15^\circ$) and gentle coefficients $(\alpha,\beta)$ (defaults $\sim 0.3$).

If the Moon is below the horizon ($h_{\rm Moon} < 0^\circ$), the constraint is waived.

### 3) Photometric Kernel (Zeropoint, Extinction, Sky, SNR)

We treat counts in photo‑electrons. For filter $m$, define a 1‑s instrumental zeropoint ($ZP_{1\rm s,m}$) such that a source of magnitude $ZP_{1\rm s,m}$ yields 1 e⁻ s⁻¹ at unit airmass.

- Extinction‑corrected rate at airmass X for a source of magnitude m:

```math

R_*(m, X) = 10^{-0.4[\,m - ZP_{1\rm s} + k_m(X-1)\,]}\quad [\mathrm{e^-\,s^{-1}}]

```

- Total source electrons in exposure time t:

```math

F_* = R_*(m,X)\,t.

```

- Sky background per pixel (electrons) uses a twilight sky model or `rubin_sim.skybrightness` if available. Given sky surface brightness $\mu_{\rm sky}$ [mag/arcsec²], pixel scale p [arcsec/px], and airmass X,

```math

B_{\rm px} = 10^{-0.4[\,\mu_{\rm sky} - ZP_{1\rm s} + k_m(X-1)\,]} p^2 t.

```

- Effective noise pixels for a Gaussian PSF with FWHM $\theta$ and pixel scale p (arcsec/px) use

```math

n_{\rm pix} \approx 4\pi\sigma_{\rm pix}^2, \qquad \sigma_{\rm pix}=\frac{\theta}{2\sqrt{2\ln 2}\,p}.

```

- SNR for a point source:

```math

\mathrm{SNR} = \frac{F_*}{\sqrt{F_* + n_{\rm pix}\big(B_{\rm px} + \mathrm{RN}^2\big)}},

```

with RN the read‑noise (e⁻). This SNR is used for feasibility checks and for the 5σ depth below.

### 4) 5σ Depth (m_5) (Twilight‑aware)

Solve SNR (=5) for the magnitude that just reaches 5σ in exposure t.

Using $F_*(m) = 10^{-0.4[\,m - ZP_{1\rm s} + k(X-1)\,]} t$ and the SNR above:

```math

m_5 \approx ZP_{1\rm s} - k(X-1) - 2.5\log_{10}\left(\frac{5}{t}\sqrt{n_{\rm pix}\left(B_{\rm px} + \mathrm{RN}^2\right)}\right),

```

which is accurate in the background‑dominated regime (typical in twilight).

The planner adds gentle Sun/Moon penalties to $m_5$ in bright conditions and falls back to redder filters if needed.

Filter feasibility rule (per candidate/time): select the first filter m for which $m_5 - m_{\rm target} \ge \Delta m_{\rm margin}$ (default margin 0.3 mag). If magnitudes are unavailable, the code uses a conservative band order favoring r/i/z at high sky brightness.

### 5) Saturation Guard (Central‑Pixel Model)

To avoid CCD blooming, we approximate the central‑pixel electrons for a Gaussian PSF:

- Peak pixel fraction:

```math

f_{\rm peak} \approx \frac{1}{2\pi\sigma_{\rm pix}^2}\underbrace{p^2}_{\text{pixel area}}, \qquad \sigma_{\rm pix}=\frac{\theta}{2\sqrt{2\ln 2}\,p}.

```

- Central pixel electrons: $N_{\rm cen} \approx f_{\rm peak}F_*$.

If $N_{\rm cen} > N_{\rm sat}$ (default $N_{\rm sat}\sim 1\times 10^5$ e⁻), the planner shortens the exposure.

Because $F_* \propto t$, the bright‑limit magnitude that saturates scales as

```math

m_{\rm sat}(t) = m_{\rm sat}(t_0) + 2.5\log_{10}\left(\frac{t}{t_0}\right),

```

so 1 s vs 15 s shifts the r‑band bright limit by $\approx 2.9$ mag, enabling very short twilight snaps to avoid saturation.

### 6) Priority Scoring (Hybrid → LC)

For each SN we track:

- $N_{\rm det}$: number of detections

- $T_{\rm exp}$: accumulated exposure time

- $|\mathcal{F}|$: number of distinct filters used

Define goal progress for hybrid and LC stages:

```math

P_{\rm hybrid} = \max\Bigg( \frac{N_{\rm det}}{2}, \frac{T_{\rm exp}}{300\,\mathrm{s}} \Bigg) ;;;\text{and};;;\

P_{\rm LC} = \max\Bigg( \frac{N_{\rm det}}{5}, \frac{T_{\rm exp}}{300\,\mathrm{s}} \Bigg).

```

Clamp to $[0,1]$. The need score at a candidate time is

```math

S_{\rm need} = \begin{cases}

1 - P_{\rm hybrid}, & \text{if strategy = hybrid and hybrid not met} \\

\mathbf{1}_{\rm Ia}(1 - P_{\rm LC}), & \text{if hybrid met (Ia escalates)} \\

0, & \text{if hybrid met and non-Ia}

\end{cases}

```

and the overall sort key is $(S_{\rm need}, \sin h)$, i.e., need first, then altitude.

### 7) SIMLIB Export (SNANA)

When `--simlib-out` is provided, the planner writes `S:` rows with the following fields per visit:

- `MJD` — mid‑exposure JD − 2400000.5

- `BAND` — LSST band

- `GAIN` — electrons/ADU (if using electrons, set gain consistently)

- `READNOISE` — e⁻ (per pixel)

- `SKYSIG` — e⁻/px (per exposure)

- `PSF_FWHM` — arcsec

- `ZPTAVG` — effective zeropoint for the exposure (1‑s ZP adjusted by extinction and any instrumental constants)

- `MAG` — -99 (simulation; true flux provided elsewhere)

The above are computed from the photometric kernel (Sections 3–5). The SIMLIB is minimal but sufficient for fast SNANA simulations.
---

## Outputs
- Per‑SN plan — CSV: date, window, chosen time, altitude, filters, exposure settings, and a detailed time budget (slew/readout/filter‑change)
- Night summary — CSV: counts of visible vs planned targets, cumulative time per window
- SIMLIB — optional SNANA SIMLIB for the planned visits

---

## Module Overview
- `config.py` — `PlannerConfig` (site, filters, caps, overheads, photometry)
- `priority.py` — state tracking & hybrid→LC escalation (Ia keep priority; non‑Ia drop after quick color)
- `scheduler.py` — nightly visibility, scoring, per‑window scheduling, time accounting
- `astro_utils.py` — twilight windows, airmass, Sun/Moon geometry, slews
- `filter_policy.py` — 5σ/m5 heuristic with Sun/Moon penalties and red‑band fallback
- `photom_rubin.py` — Rubin‑tuned photometric kernel & saturation guard
- `sky_model.py` — twilight/dark sky providers; optional `rubin_sim.skybrightness`
- `simlib_writer.py` — minimal SNANA SIMLIB exporter
- `io_utils.py` — robust CSV parsing; RA/Dec/discovery inference
- `main.py` — CLI wrapper

---

## Documentation
All modules include NumPy-style docstrings detailing parameters, return values,
and algorithmic behavior for reference.

---

## Known Limitations / Roadmap
- Heuristic $m_5$ during twilight is conservative; prefer `rubin_sim` sky brightness when available
- Moon model uses graded geometric separation; empirical tuning welcome
- Saturation threshold default is moderately conservative; tighten if blooming artifacts matter
- Single‑filter visits maximize breadth; enable multi‑filter only if caps and filter‑change budgets allow

---

## Parameter Defaults

Below is a compact table summarizing key default values, with direct links to their sources for easy reference.

| Parameter | Default Value | Source & Link |
| --- | --- | --- |
| Sun altitude for twilight | −18° to 0° | — |
| Minimum target altitude | 20° | — |
| Hybrid detections threshold | 2 detections or 300 s | — |
| Light‑curve (LC) threshold | 5 detections or 300 s | — |
| Slew small‑angle threshold | ≤ 3.5° ≈ 4 s | [Rubin slew & settle specs](https://en.wikipedia.org/wiki/Vera_C._Rubin_Observatory) |
| Readout time | 2 s per exposure | [LSST read/write timing](https://www.lsst.org/scientists/keynumbers) |
| Filter change overhead | 120 s | — (typical value in OpSim‑style planners) |
| Airmass formula | Kasten–Young (1989) | [Kasten & Young, Appl. Opt. 28, 4735–4738 (1989)](https://doi.org/10.1364/AO.28.004735) |
| Horizon airmass (~90°) | ≲ 38 | [Wiki Kasten–Young accuracy](https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Kasten_and_Young) |
| Carousel capacity (filters) | 5 | — |
| Saturation threshold | ~1 × 10⁵ e⁻ (configurable) | — |

---

## References & Notes
1. Kasten, F., & Young, A. T. (1989). Revised optical air mass tables and approximation formula. *Applied Optics*, 28(22), 4735–4738. [DOI: 10.1364/AO.28.004735](https://doi.org/10.1364/AO.28.004735)
2. Wikipedia: Air mass (astronomy) — airmass ≈38 at horizon per Kasten–Young formula
3. Wikipedia: Vera C. Rubin Observatory — slew of 3.5° and settle within 4 s
4. LSST documentation: readout & shutter/open timings (~2 s), filter/exposure overhead generalizations
5. LSST observing strategy notes / OpSim references: filter exchange timings, slews etc.
