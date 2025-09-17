# LSST Twilight Planner

Modular planner for scheduling **Vera C. Rubin Observatory (LSST)** twilight observations of supernovae (SNe).
Optimized for fast, shallow snaps in bright sky, with options to export **SNANA SIMLIBs** for downstream simulations. A minimum inter-exposure spacing of 15 s is enforced. Readout overlaps with slewing, so the natural inter-visit gap is max(slew, readout) + cross-filter-change. If that gap is shorter, idle guard time is inserted before the next exposure. Guard time is accounted for prior to window cap checks and reported in per-row and per-window summaries.

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
    --evening-twilight 18:00 --morning-twilight 05:00 \
    --filters griz --exp g:5,r:5,i:5,z:5 \
    --min_alt 20 --evening_cap auto --morning_cap auto \
    --per_sn_cap 120 --max_sn inf --strategy hybrid \
    --hybrid-detections 2 --hybrid-exposure 300 \
    --simlib-out results/night.SIMLIB --simlib-survey LSST_TWILIGHT
```

``--max_sn`` accepts numbers or "inf"/"none"/"unlimited"; the default is
unlimited and lets per-window time caps govern the final count.

To force LSST-only light curves for all SNe:

```bash
python -m twilight_planner_pkg.main --csv your.csv --out results \
    --start 2024-01-01 --end 2024-01-07 \
    --lat -30.2446 --lon -70.7494 --height 2663 \
    --strategy lc --lc-detections 5 --lc-exposure 300
```

To restrict the input catalog to Type Ia-like objects, add `--only-ia`:

```bash
python -m twilight_planner_pkg.main --csv your.csv --out results \
    --start 2024-01-01 --end 2024-01-07 --lat -30.2446 --lon -70.7494 --height 2663 \
    --filters griz --exp g:5,r:5,i:5,z:5 --only-ia
```

---

## Priority Modes

1. **Discovery‑optimized** — maximize breadth with minimal repeats
2. **Hybrid (default)** — quick color (≥2 detections in ≥2 filters or ≥300 s total).
   - If Type Ia: escalate to the light‑curve goal
   - Else: drop priority (reallocate time)
   - Low‑z boost: among candidates still needing observations, lower‑redshift
     SNe are gently favored as a tie‑breaker (configurable; on by default)
3. **LSST‑only light curves** — pursue a full LC for every SN (≥5 detections or ≥300 s across ≥2 filters)
4. **`unique_first`** — maximize distinct SNe per night; repeats get a negative
   score after the first detection (dropped before capping). After
   ``unique_lookback_days`` the optional ``unique_first_resume_score`` can revive
   targets. Default lookback is 999 d.

`unique_first_fill_with_color` is a placeholder knob for a future second pass that would add color to unique-first selections.

The tracker records every visit's color group and computes a unified filter
bonus blending cadence pressure, per-filter cosmology weights, and any
blue/red deficit so missing colors are filled early.

### Low‑Redshift Prioritization (Hybrid)

- When running the default `hybrid` strategy, the planner applies a mild
  multiplicative boost to positive priority scores based on redshift, favoring
  lower‑z objects. If no redshift is available, there is no change.
- Controlled by `PlannerConfig`:
  - `redshift_boost_enable` (default: `True`)
  - `redshift_low_ref` (default: `0.1`) — ramp ends here; z ≥ this gets no boost
  - `redshift_boost_max` (default: `1.2`) — max ×1.2 at z≈0
  - `redshift_column` (optional) — force the column name if auto‑detection fails

Input catalogs with columns like `redshift`, `z`, `zspec`, `zphot`, `zbest`, or
`host_z` are auto‑recognized; a normalized `redshift` column is created.

### Cadence constraint (per-filter)

- Revisit spacing is enforced **per filter**, not per supernova.
- A same-band revisit is blocked until `cadence_days_target - cadence_jitter_days` days have
  elapsed since that band was last observed.
- First-time observations in a band always pass the gate, enabling quick colors.
- A Gaussian “due-soon” bonus nudges bands whose last visit is near the target cadence
  without hard-blocking other filters.
- Nightly summaries report per-filter cadence compliance via
  `cad_median_abs_err_by_filter_csv` and `cad_within_pct_by_filter_csv`, with overall
  aggregates `cad_median_abs_err_all_d` and `cad_within_pct_all`.

---

## Notebook Example

```python
from twilight_planner_pkg.config import PlannerConfig
from twilight_planner_pkg.scheduler import plan_twilight_range_with_caps

cfg = PlannerConfig(lat_deg=-30.2446, lon_deg=-70.7494, height_m=2663)
# Default hybrid mode includes a gentle low‑z boost; to disable:
# cfg.redshift_boost_enable = False
plan_twilight_range_with_caps(
    '/path/to/your.csv',
    '/tmp/out',
    '2024-01-01',
    '2024-01-07',
    cfg,
    run_label='hybrid_strategy',
)
```

To select only Type Ia-like objects programmatically:

```python
cfg.only_ia = True
```

If ``run_label`` is omitted, ``"hybrid"`` (the default priority strategy) is
used for the filename prefix.

`PlannerConfig` highlights:
- `priority_strategy`: "hybrid" (default), "lc", or "unique_first"
- `hybrid_detections` / `hybrid_exposure_s`: quick‑color thresholds (defaults: ≥2 detections across ≥2 filters or ≥300 s total)
- `lc_detections` / `lc_exposure_s`: LC thresholds (defaults: ≥5 detections or ≥300 s across ≥2 filters)
- Default escalation: once hybrid is met, cached `SN_type_raw` from the CSV decides whether to escalate (Ia) or deprioritize (non‑Ia)
- Low‑z boost (hybrid): `redshift_boost_enable=True` (default), `redshift_low_ref=0.1`,
  `redshift_boost_max=1.2`, and optional `redshift_column`

---

## How It Works (Algorithm)

### Inputs & Pre‑processing
- **Inputs** — candidates CSV, output directory, UTC start/end dates, and a `PlannerConfig` (site, filters, caps, overheads)
- **Columns** — RA, Dec, discovery date (ISO or MJD), name, and type are auto‑detected if not specified
- **Derived** — `RA_deg`, `Dec_deg`, `discovery_datetime` (UTC), `Name`, `SN_type_raw`,
  and (if present) a numeric `redshift` column are normalized/added

### Eligibility
- A target must rise above `min_alt_deg` (default 20°) at some sampled time
- Supernovae without a valid `discovery_datetime` are skipped
- Visibility window lasts `ceil(1.2 × days)` after discovery where `days`
  comes from `typical_days_by_type` or a default fallback

> ### Key selection criteria recap:
>
> - A supernova must have a valid discovery date within the configured observation window.
> - Must be observable at altitude ≥ `min_alt_deg` (default 20°) during at least one twilight window where the Sun altitude lies between `twilight_sun_alt_min_deg` and `twilight_sun_alt_max_deg` (defaults −18° to 0°).
> - Eligibility duration after discovery is scaled from a `typical_days_by_type` value (or a default if type unknown) and multiplied by 1.2 to provide a buffer.
> - Observation times must also pass the Moon separation rule, which applies a filter-dependent minimum separation, waived if the Moon is below the horizon.

### Twilight Windows & Best Time
- Twilight windows are spans with Sun altitude
```math
 h_\odot \in [\texttt{twilight\_sun\_alt\_min\_deg},\texttt{twilight\_sun\_alt\_max\_deg})]
```
(defaults −18° to 0°)
- Sample each window every `twilight_step_min` minutes
- For each SN, choose the time of maximum altitude that passes altitude and Moon constraints (below)

### Selection & Scheduling
- Rank visible SNe by need score (how far from meeting the current goal) and altitude
- Rank by priority/altitude, drop non‑positive scores for ``unique_first``, then
- apply a stratified cap per twilight window based on ``max_sn_per_night``
  (default infinity).
- Within each window, schedule via greedy nearest‑neighbor on great‑circle distance
- Enforce window caps (`morning_cap_s`, `evening_cap_s`, default "auto" uses window duration) and per‑SN cap (`per_sn_cap_s`); trim filters greedily to fit

> ### Prioritization strategy recap:
>
> - Hybrid strategy (default): Aim for quick color (≥ 2 detections in ≥ 2 filters or ≥ 300 s total). Once met:
>  - Type Ia SNe escalate to the LSST-only light-curve goal (≥ 5 detections or ≥ 300 s across ≥ 2 filters).
>  - Non-Ia SNe drop to zero priority, freeing time for other targets.
> - LSST-only light curve strategy: Every SN is pursued until the LC goal is met.
> - Unique-first strategy: Observe each SN at most once per night to maximize distinct targets.
> - Candidates are ranked nightly first by need score (how far from meeting the active goal) and then by their maximum altitude within the twilight window.
> - Scheduling within each window uses a greedy nearest-neighbor approach on sky position to minimize slew time, constrained by per-SN and per-window time caps.

### Slews & Overheads
- Two‑regime slew: small moves ≤3.5° take ≈4 s; larger moves add $t_{\rm slew} \approx 4\,\mathrm{s} + \frac{\max(0,\Delta\theta-3.5^\circ)}{5.25^\circ/\mathrm{s}} + 1\,\mathrm{s}_{\rm settle}$
- Readout: 2 s per exposure; filter change: 120 s per swap
- Per‑SN budget: $T_{\rm SN} = t_{\rm slew} + \sum_{f \in \mathcal{F}}(t_{\rm exp}(f) + t_{\rm read}) + N_{\rm changes}\,t_{\rm filt}$

### Filters & Exposure Assignment
- Filters are requested in priority order and truncated if adding another
  would exceed `per_sn_cap_s`
- Planning continues even if requested filters exceed
  `carousel_capacity`, but a warning is issued
- `sun_alt_policy` is enforced: filters forbidden at the current Sun altitude are skipped.
- Optional `sun_alt_exposure_ladder` overrides default exposure times based on Sun altitude.

### Non-linearity & Saturation Policy

- Hard pixel saturation cap: **100&nbsp;ke⁻/pixel**
- Non-linear warning region: **80–100&nbsp;ke⁻/pixel**
- The planner flags exposures with `warn_nonlinear` when predicted charge
  falls in the warning band and sets `saturation_guard_applied` when the
  exposure is shortened to respect the hard cap.
- Exposure capping considers both source and sky background electrons; whichever is higher drives the reduction.

#### Dynamic twilight sky in capping

The planner now applies the Sun-altitude–aware sky brightness when computing
safe exposure times, not just when reporting sky values.  In bright twilight
the background per pixel rises rapidly, so exposures are automatically
shortened before detector saturation.  The hard 100 ke⁻ cap and 80–100 ke⁻
warning band are evaluated against the maximum of source and sky electrons.
Any configured `sun_alt_exposure_ladder` sets an initial guess; the capping
logic then enforces detector safety.

#### Discovery-magnitude fallback (default enabled)

When explicit per-band source magnitudes are missing in the input catalog, the
planner now falls back to the catalog's discovery magnitude to drive saturation
guard. This prevents overly long exposures in twilight even without per-band
photometry.

- If the catalog contains a `discoverymag` column, the scheduler builds a
  per-band map as follows (configurable via `PlannerConfig`):
  - `discovery_policy = "atlas_transform"` (default): if `discmagfilter` is
    ATLAS `cyan`/`orange` (`c`/`o`), convert to r-band using a simple
    color relation (assumed `g-r`, default 0.0). Copy that r to other bands with
    a small safety margin (default 0.2 mag). If `discmagfilter` is `r`/`g`, use
    it directly (with `g→r` using the same assumed color). Unknown filters fall
    back to a conservative copy with margin.
  - `discovery_policy = "copy"`: copy `discoverymag` to all planned bands with
    the safety margin.
- Knobs:
  - `use_discovery_fallback` (default `True`)
  - `discovery_assumed_gr` (default `0.0`)
  - `discovery_margin_mag` (default `0.2`)

This fallback is applied only when a discovery-magnitude column exists in the
input. Otherwise the behavior is unchanged.

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

Default dark‑sky surface brightness values (u:23.05, g:22.25, r:21.20, i:20.46, z:19.61, y:18.60 mag/arcsec²) follow SMTN‑002 zenith estimates,
and the pixel scale defaults to 0.2 arcsec/px per Rubin Observatory specifications. Prefer `rubin_sim.skybrightness` when available; the
`twilight_delta_mag=2.5` offset is a legacy fallback. Airmass $X$ is computed via the Kasten–Young (1989) approximation.

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

### 5) Saturation Guard (Central‑Pixel Sum)

To avoid CCD blooming we evaluate the total electrons in the brightest pixel as the sum of three components:

- SN point source central pixel: $N_{\rm src,cen} = f_{\rm cen}(\theta, p)\,F_*$ with a Gaussian PSF; we use

```math
f_{\rm cen} = \mathrm{erf}\!\left(\frac{p}{2\sqrt{2}\,\sigma}\right)^2, \qquad \sigma=\frac{\theta}{2.355}.
```

- Host galaxy local surface brightness (observed frame) as electrons per pixel:

```math
N_{\rm host/pix} = 10^{-0.4 (\mu_{\rm host,obs} - ZP_{\rm e})}\, p^2.
```

- Sky background per pixel:

```math
N_{\rm sky/pix} = 10^{-0.4 (\mu_{\rm sky} - ZP_{\rm e})}\, p^2.
```

The pixel is considered saturated if the sum exceeds the full‑well threshold:

```math
N_{\rm tot} = N_{\rm src,cen} + N_{\rm host/pix} + N_{\rm sky/pix} > N_{\rm sat}.
```

We shorten the exposure time linearly to bring $N_{\rm tot}$ under $N_{\rm sat}$ and flag frames exceeding a non‑linearity warning threshold (default 80 ke⁻). Defaults for $N_{\rm sat}$ are 100 ke⁻, adjustable via configuration.

If only rest‑frame host surface brightness is available, we apply Tolman dimming (and optional K‑correction) to obtain the observed value:

```math
\mu_{\rm host,obs} = \mu_{\rm host,rest} + 2.5\log_{10}(1+z)^4 + K(z) = \mu_{\rm host,rest} + 10\log_{10}(1+z) + K(z).
```

An optional compact host knot can be included as an extra point‑like component with its own central‑pixel fraction.

Default host SB (when per‑target host inputs are absent): the planner can apply a conservative r‑band typical value of \(\mu_{\rm host}\approx22\,\mathrm{mag/arcsec^2}\) (reflecting the 21–23 range seen in SDSS target selection and Freeman’s disk central SB after band conversion). This fallback is enabled by default and can be customized or disabled via `PlannerConfig.use_default_host_sb` and `PlannerConfig.default_host_mu_arcsec2_by_filter`.

### Discovery‑Magnitude Fallback (ATLAS c/o to r with Color Priors)

When per-band source magnitudes are missing, the planner can estimate them from
the catalog discovery magnitude to drive saturation capping. The default policy
(`discovery_policy = "atlas_priors"`) implements a conservative, physics-guided
fallback using ATLAS `cyan`/`orange` color terms and SN color priors.

1) Field‑star linear color terms (per field/night, if available):

```
(r - c)_star = alpha_c + beta_c * (g - r)_star,
(r - o)_star = alpha_o + beta_o * (g - r)_star.
```

Defaults (if no per‑field fit): `alpha_c=0, beta_c=-0.47`, `alpha_o=0, beta_o=+0.26`.

2) Discovery band → r band (SN): with discovery filter `f_disc in {c,o,g,r}`

```
r_SN ≈ m_disc +
  { alpha_c + beta_c*(g-r)_SN,  if f_disc=c
    alpha_o + beta_o*(g-r)_SN,  if f_disc=o
    - (g-r)_SN,                 if f_disc=g
    0,                          if f_disc=r }
```

To avoid saturation we choose the endpoint of `(g-r)_SN` that minimizes
`r_SN` given the coefficient sign: if the coefficient in front of `(g-r)` is
positive, choose the lower bound; if negative, the upper bound. Unknown
`f_disc` falls back to a conservative copy with a small safety margin.

3) Extrapolate from r to other bands using color‑prior intervals:

```
g_SN ≈ r_SN + (g - r)_prior,
i_SN ≈ r_SN - (r - i)_prior,
z_SN ≈ i_SN - (i - z)_prior,
y_SN ≈ z_SN - (z - y)_prior - Δy.
```

For saturation safety, the endpoint of each color prior is chosen to minimize
the target‑band magnitude (i.e., brighten the band). An extra y‑band safety
margin `Δy` (default 0.25 mag) addresses larger uncertainty.

4) Color priors (Ia‑like near peak; AB magnitudes):

```
(u - g) ∈ [0.3, 1.0],
(g - r) ∈ [-0.25, +0.15],
(r - i) ∈ [-0.15, +0.25],
(i - z) ∈ [-0.10, +0.20],
(z - y) ∈ [-0.05, +0.30].
```

For unknown type or non‑Ia, the chosen endpoint is widened by
`discovery_non_ia_widen_mag` (default 0.1 mag) in the brightening direction.
Priors and margins are configurable via `PlannerConfig`:

- `discovery_color_priors_min`, `discovery_color_priors_max`
- `discovery_non_ia_widen_mag`
- `discovery_y_extra_margin_mag`

This fallback only affects the saturation guard; science photometry is
unchanged. Per‑band magnitudes in the input take precedence when present.

By default, if a `discoverymag` column exists but per‑band magnitudes cannot be
inferred for any target, the planner raises a `ValueError`
(`discovery_error_on_missing=True`). This ensures the saturation guard is never
silently bypassed. Set this flag to `False` only if you explicitly want to
allow baseline exposures in rare cases where discovery metadata are unusable.

### 6) Priority Scoring (Hybrid → LC)

The planner keeps a small per‑SN history (detections, total exposure, filters used,
time of last visits per filter). Nightly ranking across SNe uses a binary need
flag rather than a continuous progress score:

- Hybrid stage: need = 1 until either (i) detections ≥ `hybrid_detections` with
  ≥2 distinct filters, or (ii) `exposure_s` ≥ `hybrid_exposure_s`. Once met:
  - If the SN is Type Ia (case‑insensitive match on `SN_type_raw`), it escalates
    to the LC stage and keeps need = 1 there until the LC goal is met.
  - Otherwise it drops to need = 0 (deprioritized for the night).
- LC stage: need = 1 until either detections ≥ `lc_detections` or
  (`exposure_s` ≥ `lc_exposure_s` and ≥2 distinct filters); then need = 0.
- Unique‑first strategy: need = 1 until the first detection, then need = −1 so it
  is dropped before caps. After `unique_lookback_days`, an optional
  `unique_first_resume_score` can re‑enable repeats.

Across‑SN nightly sorting uses `(priority_score, max_alt_deg)` with need first.

Filter choice within a visit is cadence‑ and band‑aware:

- Cadence gate (per filter): a same‑band revisit is allowed only if days since
  last use ≥ `cadence_days_target − cadence_jitter_days`, or if that band has
  never been observed for this SN. Bands failing the gate may defer the target.
- Cadence bonus: among the gated bands, a Gaussian “due‑soon” bonus centered at
  `cadence_days_target` with width `cadence_bonus_sigma_days` and weight
  `cadence_bonus_weight` nudges selection; a never‑before‑seen band receives
  `cadence_first_epoch_bonus_weight` instead of the Gaussian.
- Cosmology and colour boost: bands are multiplied by `cosmo_weight_by_filter`
  (default g>r>i>z>y) and by a colour‑balance boost
  `1 + alpha × deficit`, where `alpha = color_alpha` and `deficit` counts how
  many visits are missing in the relevant colour group within the last
  `color_window_days` relative to `color_target_pairs`.
  Colour groups are BLUE = {g, r} and RED = {i, z, y}. If only one colour group
  has been seen so far, the first band from the other group can receive an extra
  `first_epoch_color_boost`.
- Two‑band visits: when `max_filters_per_visit ≥ 2`, the second band prefers the
  opposite colour group to build quick colour.

If cadence is disabled, the first band is chosen to accelerate colour in Hybrid
(prefer a band not yet used); once escalated, the reddest allowed band is
preferred, with a small bias for the current carousel filter to avoid a swap.
All choices respect `sun_alt_policy`, Moon‑separation checks, and per‑SN caps.

#### Mathematical form

Let thresholds be `H_det = hybrid_detections`, `H_exp = hybrid_exposure_s`,
`L_det = lc_detections`, `L_exp = lc_exposure_s`. Let `F` be the set of filters
used so far and `|F|` its size. Define

```math
\text{met}_\mathrm{H} := \mathbf{1}\Big( (N_{\rm det} \ge H_{\rm det} \wedge |\mathcal F| \ge 2) 
\;\vee\; (T_{\rm exp} \ge H_{\rm exp}) \Big),\\
\text{met}_\mathrm{L} := \mathbf{1}\Big( (N_{\rm det} \ge L_{\rm det}) 
\;\vee\; (T_{\rm exp} \ge L_{\rm exp} \wedge |\mathcal F| \ge 2) \Big).
```

- Strategy = hybrid, with Ia escalation:

```math
S = \mathbf{1}\big(\neg\,\text{met}_\mathrm{H}\big)
\;\vee\; \Big( \mathbf{1}(\text{Ia}) \wedge \mathbf{1}\big(\text{met}_\mathrm{H}\big) \wedge \mathbf{1}\big(\neg\,\text{met}_\mathrm{L}\big) \Big),
```

so `S ∈ {0,1}` and equals 1 if Hybrid not yet met, or if Hybrid is met and the
SN is Ia but the LC goal is still unmet; otherwise `S=0`.

- Strategy = lc (LSST‑only light curves):

```math
S = \mathbf{1}\big(\neg\,\text{met}_\mathrm{L}\big).
```

- Strategy = unique_first:

```math
S = \begin{cases}
1, & N_{\rm det}=0,\\[4pt]
S_{\rm resume}, & N_{\rm det}>0\ \wedge\ (\text{now}-t_{\rm last}) > D_{\rm lookback},\\[4pt]
-1, & \text{otherwise,}
\end{cases}
```

with `D_lookback = unique_lookback_days` and `S_resume = unique_first_resume_score`.
Across SNe, sorting uses `(S, max_alt_deg)`.

Filter selection uses a gated, cadence‑ and colour‑aware utility. For filter `f`
let `Δ_f` be days since last observation in `f` (undefined if never observed).

- Gate per filter:

```math
G_f = \mathbf{1}\Big( \Delta_f\ \text{undefined}\ \vee\ \Delta_f \ge \max\{0, D_\mathrm{tgt}-J\} \Big),
```

with `D_tgt = cadence_days_target` and `J = cadence_jitter_days`.

- Cadence bonus:

```math
C_f = \begin{cases}
w\,\exp\!\Big( -\tfrac12\big(\tfrac{\Delta_f-D_\mathrm{tgt}}{\sigma}\big)^2 \Big), & \Delta_f\ \text{defined},\\[6pt]
w_{\rm first}, & \text{otherwise,}
\end{cases}
```

with `w = cadence_bonus_weight`, `σ = cadence_bonus_sigma_days`, and
`w_first = cadence_first_epoch_bonus_weight`.

- Colour/cosmology boost. With BLUE = {g,r}, RED = {i,z,y}, define counts in the
recent window `W = color_window_days`:

```math
V_\mathrm{blue}, V_\mathrm{red} := \#\;\text{visits in }[\text{now}-W,\,\text{now}]\ \text{by group}.
```

Let `T = color_target_pairs`. For a filter `f` in group `\mathcal G(f) \in \{\mathrm{blue},\mathrm{red}\}`,

```math
d(f) = \max\big(0,\, T - V_{\mathcal G(f)}\big),\qquad B_{\rm color}(f) = 1 + \alpha\, d(f),
```

with `α = color_alpha`. Let `w_f = cosmo_weight_by_filter[f]`. If only one
colour group has been seen so far, the first band from the other group receives
an additional nudge `n(f) = \max\{1,\,\texttt{first_epoch_color_boost}\}`; else `n(f)=1`.

- Combined utility and choice:

```math
U(f) = C_f\; w_f\; B_{\rm color}(f)\; n(f),\qquad f^* = \underset{f\in\mathcal A\cap\{G_f=1\}}{\arg\max}\; U(f),
```

where `\mathcal A` is the allowed set after Sun/Moon and policy checks. If a
preference `first_filter` is available and passes the gate, it is kept;
otherwise `f*` is used. When `max_filters_per_visit ≥ 2`, the second band prefers
the opposite colour group to `f*`; if none, the next‑best `U(f)` is used.

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

---

## Outputs
- Per‑SN plan — CSV (`lsst_twilight_plan_<run_label>_<start>_to_<end>.csv`): date, window, chosen time, altitude, filters, exposure settings, and a detailed time budget (slew/readout/filter‑change). Represents the **best‑in‑theory** schedule keyed to each target's `best_time_utc`; times may overlap across different SNe and do not reflect the serialized on‑sky order.
- True sequence CSV — `lsst_twilight_sequence_true_<run_label>_<start>_to_<end>.csv`: **true, non‑overlapping execution order** within each twilight window. Visits are packed as soon as the previous one ends (ignoring `best_time_utc` slack); the original preference is recorded as `preferred_best_utc`. Columns include `order_in_window`, `sn_start_utc`, `sn_end_utc`, and `filters_used_csv`. One row per SN visit (multi‑filter visits are a single row).
- Night summary — CSV (`lsst_twilight_summary_<run_label>_<start>_to_<end>.csv`): counts of visible vs planned targets, cumulative time per window
- SIMLIB — optional SNANA SIMLIB for the planned visits

In all filenames above, ``<run_label>`` defaults to ``hybrid`` to reflect the
default priority strategy.

---

## Module Overview
- `config.py` — `PlannerConfig` (site, filters, caps, overheads, photometry)
- `priority.py` — per‑SN visit logging with unified cadence/cosmology/color bonus; hybrid→LC escalation (Ia keep priority; non‑Ia drop after quick color)
- `scheduler.py` — nightly visibility, scoring, per‑window scheduling, palette rotation, swap‑cost amortization, and time accounting; applies `sun_alt_policy` and optional exposure ladder
- `astro_utils.py` — twilight windows, airmass, Sun/Moon geometry, slews
- `filter_policy.py` — 5σ/m5 heuristic with Sun/Moon penalties; g‑band only in the darkest twilight with extra headroom
- `photom_rubin.py` — Rubin‑tuned photometric kernel with source+sky saturation guard
- `sky_model.py` — twilight/dark sky providers with Sun-altitude brightening; optional `rubin_sim.skybrightness`
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
| Sun altitude for twilight | `twilight_sun_alt_min_deg` to `twilight_sun_alt_max_deg` (default −18° to 0°) | — |
| Minimum target altitude | 20° | — |
| Hybrid detections threshold | 2 detections or 300 s | — |
| Light‑curve (LC) threshold | 5 detections or 300 s | — |
| Slew small‑angle threshold | ≤ 3.5° ≈ 4 s | [Rubin slew & settle specs](https://en.wikipedia.org/wiki/Vera_C._Rubin_Observatory) |
| Readout time | 2 s per exposure | [Rubin Observatory key numbers](https://www.lsst.org/scientists/keynumbers) |
| Filter change overhead | 120 s | [DMTN-065](https://dmtn-065.lsst.io) |
| Inter-exposure minimum | `inter_exposure_min_s = 15 s` | thermal/operational margin |
| Pixel scale | 0.2 arcsec/px | [Rubin Observatory key numbers](https://www.lsst.org/scientists/keynumbers) |
| Site location (lat, lon, alt) | −30.2446°, −70.7494°, 2663 m | [Rubin Observatory key numbers](https://www.lsst.org/scientists/keynumbers) |
| Shutter open/close time | 1 s | [Rubin Observatory key numbers](https://www.lsst.org/scientists/keynumbers) |
| Read noise | ≈6 e⁻ (typ. 5.4–6.2; requirement ≤9) | [Rubin camera specs](https://www.rubinobservatory.org), [LCA-48-J](https://project.lsst.org/lsst-camera/lca-48-j) |
| Gain | 1 e⁻/ADU (measured ≈1.5–1.7; 1 acceptable per SMTN‑002) | [SMTN-002](https://smtn-002.lsst.io) |
| Dark‑sky brightness (u:g:r:i:z:y) | 23.05, 22.25, 21.20, 20.46, 19.61, 18.60 mag/arcsec² | [SMTN-002](https://smtn-002.lsst.io) |
| Airmass formula | Kasten–Young (1989) | [Kasten & Young, Appl. Opt. 28, 4735–4738 (1989)](https://doi.org/10.1364/AO.28.004735) |
| Horizon airmass (~90°) | ≲ 38 | [Wiki Kasten–Young accuracy](https://en.wikipedia.org/wiki/Air_mass_(astronomy)#Kasten_and_Young) |
| Carousel capacity (filters) | 5 | — |
| Saturation threshold | ≈1 × 10⁵ e⁻ (PTC turnoff 103 ke⁻ e2v / 129 ke⁻ ITL) | [Rubin camera specs](https://www.rubinobservatory.org) |

---

## References & Notes
1. [Rubin Observatory key numbers](https://www.lsst.org/scientists/keynumbers) — site coordinates, pixel scale, readout/shutter timing.
2. [DMTN-065: Detailed Filter Changer Timing](https://dmtn-065.lsst.io) — 120 s filter-change breakdown.
3. [SMTN-002: Expected LSST Performance](https://smtn-002.lsst.io) — sky brightness, read-noise/gain guidance.
4. [LCA-48-J](https://project.lsst.org/lsst-camera/lca-48-j) — camera read-noise requirement (≤9 e⁻).
5. Kasten, F., & Young, A. T. (1989). Revised optical air mass tables and approximation formula. *Applied Optics*, 28(22), 4735–4738. [DOI: 10.1364/AO.28.004735](https://doi.org/10.1364/AO.28.004735)
6. Wikipedia: Air mass (astronomy) — airmass ≈38 at horizon per Kasten–Young formula
7. Wikipedia: Vera C. Rubin Observatory — slew of 3.5° and settle within 4 s
8. Ivezić, Ž., et al. (2019). LSST: From Science Drivers to Reference Design and Anticipated Data Products. *ApJ*, 873, 111. [DOI: 10.3847/1538-4357/ab042c](https://doi.org/10.3847/1538-4357/ab042c)
