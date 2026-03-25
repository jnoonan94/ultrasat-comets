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
