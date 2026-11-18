# ultrasat-comets

Tools for the UltraSat OH cometary pipeline: from planning which comets
ULTRASAT will see, to measuring them in images, to water production rates.

| Part | Folder | What it does | Python |
|---|---|---|---|
| Observability simulator | `ultrasat_object_search.py`, `sbdb_ident_query.py` | Which comets fall in ULTRASAT's field of regard over a date range (JPL sb_ident) | ≥ 3.9 |
| Comet test bench | `comet_testbench/` | Finds comets in FITS images from the WCS, measures aperture photometry, writes a hand-off file per comet | ≥ 3.9 |
| OH → H2O model | `oh_h2o_model/` (`oh_h2o`) | Continuum removal, OH flux and H2O production rate from the hand-off file, built on swift_comet_pipeline | 3.11–3.12 |
| Mission parameters | `mission_parameters/*.json` | Versioned instrument numbers (Swift/UVOT now, ULTRASAT as they become known) | – |
| Future tests | [`future_tests.md`](future_tests.md) | What to run when each ULTRASAT parameter arrives, and a results log | – |

```
FITS images ─▶ comet_testbench ─▶ results/handoff/<comet>_photometry.ecsv ─▶ oh_h2o ─▶ results/oh/<comet>_water.ecsv
                                                                               ▲
                                                         mission_parameters/<instrument>.json
```

The test bench and the OH model meet only at the hand-off file (ECSV,
schema `ultrasat-comet-photometry/1`), so each can change without the other.

## Install

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e .[test]                  # simulator + test bench
pip install -e oh_h2o_model[test]       # OH -> H2O model (pulls swift_comet_pipeline)
git clone --depth 1 https://github.com/sjoset/swift_comet_pipeline ~/src/swift_comet_pipeline
export SWIFT_COMET_PIPELINE_DATA=~/src/swift_comet_pipeline/data
```

The OH model needs swift_comet_pipeline's `data/` folder, which its PyPI
release leaves out. See [`oh_h2o_model/README.md`](oh_h2o_model/README.md).

## Quick start: Swift images to water production rates

```bash
python -m comet_testbench fetch '%WIRTANEN%' --start 2018-12-10 --stop 2018-12-20 --dest data/swift
python -m comet_testbench run data/swift/*/*_sk.img.gz --config examples/swift_46p_config.json --out results
python -m oh_h2o results/handoff/46P_photometry.ecsv --mission mission_parameters/swift_uvot.json \
    --redness 0 10 --max-dt 240 --out results/oh
```

On the six 46P benchmark frames this gives Q(H2O) of 0.5–1.3 × 10²⁸ s⁻¹ in
5,000–20,000 km apertures, in line with published values for 46P in
December 2018.

## Mission parameters

Each instrument has one JSON file in `mission_parameters/` with a
semantic `version`, a `changelog`, and for every parameter a `value`,
`unit`, `status` (measured, nominal, reference, simulator, unconfirmed, TBD)
and `source`. `oh_h2o.load_mission_parameters` checks the file, and
`oh_h2o.calibration_from_mission` turns it into the OH model's calibration.
For ULTRASAT, that call refuses to run, listing what's still TBD. ULTRASAT
measures OH in the A-X (1-0) band at ~282 nm (`oh_bands: ["1-0"]`), not the
(0-0) band at 309 nm that Swift/UVW1 sees. The method is the same, with the
(1-0) fluorescence g-factor. Every result
file records the parameter version it used. How to update a file and which
tests to rerun is in [`future_tests.md`](future_tests.md).

## Licence

`oh_h2o_model/` is GPL-3.0 (see `oh_h2o_model/LICENSE`), the licence of
swift_comet_pipeline, which it imports.

## File structure

```
ultrasat-comets/
├── ultrasat_object_search.py       # Observability simulator (FOV grids, loop, post-processing, plots)
├── sbdb_ident_query.py             # JPL SBDB sb_ident API wrapper
├── LCS_nonoverlapping_grid.csv     # Low-cadence survey field centres (29 fields)
├── midplane_20{19,22}_ultrasat_cadence.npy  # Cadence arrays (not yet used)
├── comet_testbench/                # Comet search + aperture photometry + hand-off
├── oh_h2o_model/                   # OH -> H2O package (GPL-3.0), own pyproject and tests
├── mission_parameters/             # Versioned instrument parameter files (JSON)
├── future_tests.md                 # Tests to run as ULTRASAT parameters arrive
├── examples/                       # Test bench configs
└── tests/                          # Simulator + test bench tests
```

---

# Observability simulator

UltraSat is a UV space telescope in geostationary orbit (~35,785 km altitude),
modelled here with a 20° × 10° (RA × Dec) field of view. The simulator
generates a grid of FOV centres, queries JPL sb_ident for comets and NEOs in
each, applies pointing constraints (Galactic latitude, solar elongation, lunar
separation) and plots cumulative detection maps. Supported classes: JFc, ETc,
HTC, PAR, HYP, COM.

## Usage

### Basic galactic-plane survey

```python
from ultrasat_object_search import run_observability_simulator, gen_total_observability, plot_total_observability

obs_dict = run_observability_simulator(
    v_lim=18,           # Visual magnitude limit
    start='2025-01-01',
    stop='2025-12-31',
    step=12,            # Number of evenly-spaced observation epochs
    lat_range=(-30, 30),  # Galactic latitude range (degrees)
    lon_range=(0, 360),   # Galactic longitude range (degrees)
    N=5,                # Grid resolution (N×N field centers)
)
```

### Low-cadence survey (pre-defined field grid)

```python
obs_dict = run_observability_simulator(
    v_lim=18,
    start='2025-01-01',
    stop='2025-12-31',
    step=12,
    lat_range=(-30, 30),
    lon_range=(0, 360),
    N=5,
    lcs_ingest=True,    # Load field centers from LCS_nonoverlapping_grid.csv
)
```

### Single-field stare (galactic coordinates input)

```python
obs_dict = run_observability_simulator(
    v_lim=18,
    start='2025-01-01',
    stop='2025-12-31',
    step=12,
    lat_range=0,        # Galactic latitude of target (degrees)
    lon_range=180,      # Galactic longitude of target (degrees)
    N=1,
    single_field=True,
)
```

### Post-processing and visualization

```python
ra_fov, dec_fov, n_obj, coords, names, no_filt, gal_filt = gen_total_observability(obs_dict)
plot_total_observability(ra_fov, dec_fov, n_obj, coords, no_filt, gal_filt, names)
```

`gen_total_observability` applies three filter levels and returns coordinate arrays for each:
- **No filter**: all objects found
- **Galactic filter**: objects within ±30° of the galactic plane
- **Full filter**: galactic plane + solar elongation > 70° + lunar separation > 30°

---

## Configuration

Key constants are defined at the top of `ultrasat_object_search.py`:

| Constant | Value | Description |
|---|---|---|
| `ultrasat_fov` | `[20°, 10°]` | RA × Dec field of view |
| `dec_NGP` | 27.13° | Declination of North Galactic Pole |
| `ra_NGP` | 192.85° | RA of North Galactic Pole |
| `l_NCP` | 123° | Galactic longitude of North Celestial Pole |

Observer location (hardcoded in `api_query()`):

| Parameter | Value | Description |
|---|---|---|
| `lat` | 32.028140° | Geostationary orbit latitude proxy |
| `lon` | 34.898320° | Geostationary orbit longitude proxy |
| `alt` | 35785 km | Geostationary orbit altitude |

---

## Data Files

- **`LCS_nonoverlapping_grid.csv`**: 29 non-overlapping survey fields with columns `Field`, `RA`, `Dec`, `V180`, `V45`, `A_U<1`, `AU`. Used when `lcs_ingest=True`.
- **`midplane_2019_ultrasat_cadence.npy`** / **`midplane_2022_ultrasat_cadence.npy`**: Pre-computed cadence arrays for 2019 and 2022 observing seasons (not yet integrated into the main simulation loop).

---

## Known Limitations

- API calls are made sequentially; large runs (many dates × many FOVs) can be slow.
- No retry logic if a JPL API call fails or times out.
- Moon position in `gen_total_observability()` is incorrectly computed using `get_sun()` (copy-paste bug on line 429).
- Observer location assumes a fixed geostationary sub-satellite point; actual UltraSat pointing constraints are more complex.
- No `requirements.txt` is provided; see the Installation section above.

---

# Comet test bench (`comet_testbench/`)

A test bench for the OH cometary pipeline using public Swift/UVOT images. It is
instrument- and filter-agnostic: anything with a celestial WCS, an observation
time and an exposure time can be fed in; instrument specifics live in small
adapter classes (`comet_testbench/instruments.py`).

What it does, per image extension:

1. **Find comets in the field** (`field_search.find_comets`). Candidates come from
   the header `OBJECT` keyword, a user list, and a JPL sb_ident query over the
   image footprint (reusing `sbdb_ident_query.param_build`). Each candidate is
   placed with a Horizons ephemeris *for the actual observer* (Swift is
   `500@-128485`; geocentric parallax is ~1 arcmin for a comet at 0.1 au) and kept
   only if it lands on exposed pixels.
2. **Aperture photometry** (`photometry.py`) on the count-rate image: fixed
   apertures in pix, arcsec or km at the comet, and an iterative mode that
   recentres on the coma and grows the aperture until the flux converges,
   recording the growth curve. UVOT images get a pixel-level coincidence-loss
   correction (Poole et al. 2008).
3. **Hand-off to the OH/H2O package** (`oh.py`). The bench stops at aperture
   photometry plus observing geometry and writes one plain-text ECSV file per
   comet (`results/handoff/<comet>_photometry.ecsv`). Each row is one aperture
   in one image: filter, time, radius in pix/arcsec/km, count rate, background,
   calibrated flux where known, and the comet's r_h, r_dot, Delta and phase.
   Units and column descriptions sit in the commented header. Continuum
   removal, OH fluxes, the coma model and production rates all live in the
   separately developed OH/H2O package, which reads this file with
   `astropy.table.Table.read(path)` (or any CSV reader that skips `#` lines).

## Same search for simulated pointings and real images

`comet_testbench/pointing.py` turns a field-of-regard pointing into a *virtual
frame*: a WCS built from the pointing centre, roll and the 20° × 10° field the
simulator uses. Virtual frames and downlinked FITS images both go through
`find_comets`, so the method used to plan the survey is the one used on the
data. Only the input changes.

```python
from astropy.time import Time
import ultrasat_object_search as uos
from comet_testbench import search_pointings

fovs = uos.retrieve_fov_array()                      # LCS grid, "ra dec" strings
rows = search_pointings(fovs, [Time("2025-03-01")],  # one row per comet per pointing
                        vmag_lim=18, no_magnitude="none")   # = simulator's selection
```

The UltraSat observer defaults to the simulator's site (35,785 km above
32.03°N, 34.90°E) for both sb_ident and Horizons (`UltraSat` adapter). Pass
`horizons_id` once the spacecraft has a Horizons ephemeris. Differences from
`ultrasat_object_search.py` that are deliberate:

| | simulator | test bench |
|---|---|---|
| field | RA/Dec box (`fov-ra-lim`, `fov-dec-lim`) | true footprint from the WCS, then a per-pixel check |
| comets with no magnitude model | dropped (`mag-required=true`) | numbered periodic comets kept (`no_magnitude="periodic"`); `"none"` matches the simulator |
| positions | sb_ident second pass | sb_ident, or Horizons for the real observer (`refine_with_horizons`) |
| Sun / Moon / latitude cuts | applied, rows dropped | reported as flags per comet |
| JPL rate limiting | loop stops | waits for `Retry-After` and retries |

## Quick start

```bash
pip install -e .[test]

# Download UVW1 and V images of 46P/Wirtanen from HEASARC
python -m comet_testbench fetch '%WIRTANEN%' --start 2018-11-28 --stop 2018-12-31 --dest data/swift

# Find comets and measure them; writes CSVs to results/
python -m comet_testbench run data/swift/*/*_sk.img.gz --config examples/swift_46p_config.json --out results

```

Outputs: `comets.csv` (one row per comet per frame, with geometry, trail length
and growth-curve status), `fixed_apertures.csv`, `growth_curves.csv`, and the hand-off
files in `handoff/` (skip them with `--no-handoff`).

## Tests

`pytest` runs the offline suite on synthetic comet images. `pytest -m network`
runs the live checks on real Swift frames (below); downloads are cached in
`$COMET_BENCH_CACHE` (default `data/swift`, about 150 MB).

## Benchmarks

The benchmarks check that the search and photometry carry over from the
field-of-regard simulations to real images. The metrics live in
`comet_testbench/benchmarks.py` and take any `ImageFrame`, so the same calls
run on downlinked UltraSat frames once they exist. Each one has an offline test
on synthetic images (`tests/test_benchmarks.py`) and a real-data test on eight
Swift/UVOT frames of 46P and 41P in UVW1 and V
(`tests/test_benchmarks_swift.py`, frame list in `SWIFT_BENCHMARK_FRAMES`).

| # | Benchmark | Function | Real-data check |
|---|---|---|---|
| 1 | Ephemeris recovery | `ephemeris_offset` (total, along-track, cross-track) | every frame within 1.5" plus the timing allowance; median under 1.1" |
| 2 | Non-sidereal motion | `CometInField.trail_pix`, `timing_tolerance_arcsec` | 46P at 0.08 au (trail > 100 px) still found and measured; residual lies along the motion |
| 3 | Edge and crowded fields | coverage in `ApertureResult`, recentring guard | bright stars 20 to 35" from 46P do not capture the recentring |
| 4 | Faint-limit completeness | `inject_coma`, `completeness_curve`, `fifty_percent_rate` | 1/ρ comae injected into real 41P backgrounds; 50% point within 3x of the Poisson limit, false positives ≤ 10% |
| 5 | Aperture stability | `aperture_stability` | enclosed rate never drops with radius; spread across background choices recorded |
| 6 | OH package round trip | `tests/stub_oh_package.py` | hand-off file read, units checked, UVW1 paired with V at the same km radius |
| 7 | Fields with no comet | `run` | the two Galactic Centre visits labelled `COMET41P` return nothing |

`tests/stub_oh_package.py` is a placeholder for the OH/H2O package. It reads
the hand-off file, refuses unknown schemas, missing columns and missing units,
and pairs each OH-band image with the nearest continuum image. Its "model" is a
fixed continuum scale, so it tests the interface and nothing else.

## Caveats

- UVOT zero points and flux factors in `instruments.py` are from Poole et al.
  (2008) / Breeveld et al. (2011); verify against the current CALDB.
- Each snapshot is measured separately; the comet's motion during an exposure is
  reported (`trail_pix`) but not corrected.
- The ephemeris time is the midpoint of DATE-OBS/DATE-END. When the exposure
  covers only part of that span (46P obsid 00094422001: 122 s of 276 s), the
  light-weighted time can differ, which shows up as an along-track offset
  (3" there).
- For comets at 0.1 to 0.2 au the coma fills the UVOT field, so every sky
  annulus holds coma light. A 20" aperture rate changes by 10 to 33% between
  background choices on the benchmark frames.
- Swift target names are not always reliable: two archive visits labelled
  `COMET41P` (obsids 00093389002/003) point at the Galactic Centre, and the bench
  correctly reports no comet in them.

---

# OH → H2O model (`oh_h2o_model/`)

Reads the hand-off file and returns, for every OH-band aperture: the paired
continuum rate at the same km radius, β, the OH count rate, OH flux, number
of OH molecules, and Q(H2O) (aperture-matched and whole-coma), all with 1σ
errors. The physics comes from swift_comet_pipeline. Only β and the OH flux
factor depend on the instrument, and they come from the mission parameter
file. Full details, the step-by-step mapping to swift_comet_pipeline
functions, and its known quirks are in
[`oh_h2o_model/README.md`](oh_h2o_model/README.md).


