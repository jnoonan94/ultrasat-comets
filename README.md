# ultrasat-observability

Observability simulation tool for the [UltraSat](https://www.weizmann.ac.il/ultrasat/) space telescope. Predicts which comets and small solar system bodies will fall within UltraSat's field of view across a given date range and sky region, using the [JPL Small Body Database Identification API](https://ssd-api.jpl.nasa.gov/doc/sb_ident.html).

---

## Overview

UltraSat is a UV space telescope in geostationary orbit (~35,785 km altitude) with a 20° × 10° (RA × Dec) field of view. This tool:

1. Generates a grid of FOV centers in galactic or equatorial coordinates
2. Loops through dates and FOV positions, querying JPL SBDB for observable comets and near-Earth objects
3. Applies pointing constraints (galactic latitude, solar elongation, lunar separation)
4. Produces sky maps showing cumulative detection densities

Supported small body classes: Jupiter-family comets (JFc), Encke-type comets (ETc), Halley-type comets (HTC), parabolic (PAR), hyperbolic (HYP), and unclassified comets (COM).

---

## Installation

No package installation is required beyond the dependencies below. Clone the repo and run scripts directly.

**Dependencies:**

```
astropy>=4.0
numpy
pandas
matplotlib
requests
```

Install with:

```bash
pip install astropy numpy pandas matplotlib requests
```

---

## File Structure

```
ultrasat-observability/
├── sbdb_ident_query.py             # JPL SBDB API wrapper (param building, HTTP requests)
├── ultrasat_object_search.py       # Main simulation engine (FOV grids, loop, post-processing, plots)
├── LCS_nonoverlapping_grid.csv     # Pre-computed low-cadence survey field centers (29 fields)
├── midplane_2019_ultrasat_cadence.npy  # 2019 observing cadence data
├── midplane_2022_ultrasat_cadence.npy  # 2022 observing cadence data
└── ultrasat_comets_galactic_2025.png   # Example output visualization
```

---

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

## Comet test bench (`comet_testbench/`)

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
3. **OH hand-off** (`oh.py`). Pairs UVW1 and V measurements of the same comet,
   forms `OH = UVW1 - beta * V`, converts to an OH band flux, and fits a
   production rate with any model object that implements
   `aperture_flux(production_rate, radius_km, geometry)`. The OH model itself is
   developed separately; see `examples/toy_oh_model.py` for the interface.

### Same search for simulated pointings and real images

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

### Quick start

```bash
pip install -e .[test]

# Download UVW1 and V images of 46P/Wirtanen from HEASARC
python -m comet_testbench fetch '%WIRTANEN%' --start 2018-11-28 --stop 2018-12-31 --dest data/swift

# Find comets and measure them; writes CSVs to results/
python -m comet_testbench run data/swift/*/*_sk.img.gz --config examples/swift_46p_config.json --out results

# With the OH step (beta and the OH count-rate-to-flux factor come from your calibration)
python -m comet_testbench run data/swift/*/*_sk.img.gz --beta <beta> --oh-conversion <factor> \
    --oh-model your_oh_package.module:Model
```

Outputs: `comets.csv` (one row per comet per frame, with geometry, trail length
and growth-curve status), `fixed_apertures.csv`, `growth_curves.csv`, and, when the
OH step runs, `oh_fluxes.csv` and `production_rates.csv`.

### Tests

`pytest` runs the offline suite on synthetic comet images. `pytest -m network`
runs a live check that downloads a Swift UVW1 image of 46P and confirms the
comet is found within a few pixels of the Horizons prediction.

### Caveats

- UVOT zero points and flux factors in `instruments.py` are from Poole et al.
  (2008) / Breeveld et al. (2011); verify against the current CALDB.
- Each snapshot is measured separately; the comet's motion during an exposure is
  reported (`trail_pix`) but not corrected.
- Swift target names are not always reliable: two archive visits labelled
  `COMET41P` (obsids 00093389002/003) point at the Galactic Centre, and the bench
  correctly reports no comet in them.
