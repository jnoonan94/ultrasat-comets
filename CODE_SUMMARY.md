# Code Summary & Improvement Suggestions

## Architecture

Two-module design:

```
sbdb_ident_query.py          ultrasat_object_search.py
────────────────────         ─────────────────────────────────────────
param_build()           ←─── api_query()
get_response()          ←─── api_query()
                             │
                             ├── build_fov_arrays()      ← galactic-grid mode
                             ├── retrieve_fov_array()    ← LCS CSV mode
                             ├── build_date_range()
                             │
                             └── api_loop()              ← nested date×FOV loop
                                  └── api_query()  ×N
                                       └── param_build() + get_response()

Post-processing / visualization (called separately after simulation):
  gen_total_observability(obs_dict)  →  plot_total_observability(...)
```

---

## Function Inventory

### `sbdb_ident_query.py`

| Function | Purpose | Key Inputs | Returns |
|---|---|---|---|
| `jprint(obj)` | Pretty-print JSON | dict | None |
| `param_build(date, vmag, fov_ra, fov_dec, ...)` | Build API parameter dict | date string, magnitude, FOV bounds, location | dict |
| `get_response(parameters)` | Execute HTTP GET to JPL API | param dict | `requests.Response` |
| `test_query()` | Integration smoke test (McDonald Observatory) | — | response |

### `ultrasat_object_search.py`

| Function | Purpose | Key Inputs | Returns |
|---|---|---|---|
| `build_fov_arrays(ga_lon_range, ga_lat_range, N)` | Create N×N galactic-coord FOV grid → ICRS strings | lon/lat ranges (deg), grid size | list of `"RA Dec"` strings |
| `retrieve_fov_array(debug)` | Load pre-computed LCS field centers from CSV | — | list of `"RA Dec"` strings |
| `build_date_range(start, stop, step)` | Generate evenly-spaced date list | ISO date strings, N epochs | list of `datetime` |
| `api_query(ra_lim, dec_lim, date, v_lim, fov_center)` | Single JPL API call + parse result | formatted RA/Dec limit strings, date, mag limit | `(n_results, name_list, coord_list)` |
| `sbdb_ra_rewrite(ra_min, ra_max, ra_lim)` | Format RA string for JPL API (zero-pad, negative handling) | min/max RA degrees, raw string | formatted RA string |
| `sbdb_dec_rewrite(dec_min, dec_max, dec_lim)` | Format Dec string for JPL API (`M` prefix, zero-pad) | min/max Dec degrees, raw string | formatted Dec string |
| `api_loop(fov_array, date_range, v_lim, ...)` | Nested loop over dates × FOVs, builds `obs_dict` | FOV strings, dates, mag limit | nested dict keyed by date/FOV index |
| `run_observability_simulator(...)` | Top-level orchestrator; selects protocol, calls api_loop | all config params | `obs_dict` |
| `gen_total_observability(obs_dict)` | Extract + filter coordinates from obs_dict | obs_dict | 7 arrays (FOV coords, object coords at 3 filter levels, names) |
| `plot_total_observability(...)` | Generate 6 matplotlib figures | output of gen_total_observability | None (displays figures) |

---

## Improvement Suggestions

### 1. Parallelize API Calls (highest impact)

**Problem**: `api_loop()` runs all `(date, FOV)` pairs sequentially. Each pair is an independent, I/O-bound HTTP request. A run with 29 LCS fields × 50 dates = 1,450 sequential API calls.

**Fix**: Use `concurrent.futures.ThreadPoolExecutor` to fan out the inner loop:

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def api_loop(fov_array, date_range, v_lim, debug=False, max_workers=10):
    obs_dict = {}
    tasks = [
        (i, j, date, fov)
        for i, date in enumerate(date_range)
        for j, fov in enumerate(fov_array)
    ]

    def query_one(args):
        i, j, date, fov = args
        date_str = datetime.datetime.strftime(date, '%Y-%m-%d_%H:%M:%S')
        fov_list = fov.split(' ')
        ra_cen = Angle(float(fov_list[0]) * u.deg)
        dec_cen = Angle(float(fov_list[1]) * u.deg)
        ra_lim = ...  # existing limit calculation
        dec_lim = ...
        query = api_query(ra_lim, dec_lim, date_str, v_lim)
        return i, j, date_str, fov_list, query

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(query_one, t): t for t in tasks}
        for future in as_completed(futures):
            i, j, date_str, fov_list, query = future.result()
            obs_dict.setdefault(i, {'Date': date_str, 'Data': {}})
            obs_dict[i]['Data'][j] = {
                'FOV': fov_list,
                'Number of Objects': query[0],
                'Names': query[1],
                'ICRS Coordinates': query[2],
            }
    return obs_dict
```

**Expected speedup**: ~10× with `max_workers=10`, limited by JPL API rate limits. Test with a small run first to confirm the API tolerates concurrent requests.

---

### 2. Fix Moon Position Bug

**Problem** (`ultrasat_object_search.py`, line 429):
```python
moon_coord = get_sun(date)   # BUG: this is the Sun again, not the Moon
```

The lunar separation filter silently uses the Sun's position for both the solar elongation check and the "lunar distance" check, making the lunar filter incorrect.

**Fix**:
```python
from astropy.coordinates import get_body
moon_coord = get_body('moon', date)
```

---

### 3. Refactor RA/Dec String Formatters

**Problem**: `sbdb_ra_rewrite()` (42 lines) and `sbdb_dec_rewrite()` (89 lines) manipulate strings character-by-character. They are hard to read, fragile, and contain redundant branches.

The JPL API format is `hh-mm-ss.ss` for RA and `dd-mm-ss.ss` for Dec, with `M` replacing a leading minus sign.

**Fix**: Replace with a clean formatter using `Angle.to_string()`:

```python
def format_ra_for_api(angle: Angle) -> str:
    """Format an RA Angle as hh-mm-ss.ss for the JPL SBDB API."""
    s = angle.to_string(unit=u.hour, sep='-', precision=2, pad=True)
    return s  # Angle.to_string with pad=True handles zero-padding automatically

def format_dec_for_api(angle: Angle) -> str:
    """Format a Dec Angle as [M]dd-mm-ss.ss for the JPL SBDB API."""
    s = angle.to_string(unit=u.deg, sep='-', precision=2, pad=True, alwayssign=False)
    if angle.deg < 0:
        s = 'M' + s.lstrip('-')
    return s
```

---

### 4. Remove Duplicate Import

**Problem** (`ultrasat_object_search.py`, lines 11 & 16):
```python
from astropy import units as u   # line 11
from astropy import units as u   # line 16 — exact duplicate
```

**Fix**: Delete line 16.

---

### 5. Consolidate Hard-coded Constants

**Problem**: Observer location is hard-coded inside `api_query()` (lines 138–140) rather than with the other module-level constants:

```python
lat = 32.028140
lon = 34.898320
alt = 35785
```

**Fix**: Move to the module-level constants block:

```python
# Observer location (UltraSat geostationary orbit)
OBSERVER_LAT = 32.028140   # degrees
OBSERVER_LON = 34.898320   # degrees
OBSERVER_ALT = 35785       # km
```

---

### 6. Add Retry / Rate-Limit Handling

**Problem**: `get_response()` makes a single attempt. Any transient network error or JPL rate-limit response (HTTP 429) silently propagates as a failed result.

**Fix** (simple retry loop, no extra dependencies):

```python
import time

def get_response(parameters, retries=3, backoff=2.0):
    for attempt in range(retries):
        response = requests.get("https://ssd-api.jpl.nasa.gov/sb_ident.api", params=parameters, timeout=30)
        if response.status_code == 429:
            time.sleep(backoff * (attempt + 1))
            continue
        response.raise_for_status()
        return response
    raise RuntimeError(f"API request failed after {retries} attempts")
```

---

### 7. Replace `print()` with `logging`

**Problem**: ~28 `print()` calls are scattered throughout both files. This makes it impossible to suppress debug output when calling the library programmatically.

**Fix**: Add at the top of each module:

```python
import logging
logger = logging.getLogger(__name__)
```

Then replace `print(...)` with `logger.debug(...)` or `logger.warning(...)` as appropriate. Callers can then control verbosity with `logging.basicConfig(level=logging.INFO)`.

---

### 8. Add `requirements.txt`

No dependency file currently exists.

**Fix**: Create `requirements.txt`:

```
astropy>=4.0
numpy
pandas
matplotlib
requests
```

---

## Summary Priority Table

| Priority | Issue | Effort | Impact |
|---|---|---|---|
| High | Parallelize `api_loop()` with ThreadPoolExecutor | Medium | 10–20× speedup |
| High | Fix moon position bug (line 429) | Low | Correctness |
| Medium | Refactor RA/Dec string formatters | Medium | Maintainability |
| Low | Remove duplicate `import units` | Trivial | Cleanliness |
| Low | Move observer lat/lon/alt to module constants | Trivial | Maintainability |
| Low | Add retry logic to `get_response()` | Low | Robustness |
| Low | Replace `print()` with `logging` | Low | Usability |
| Low | Add `requirements.txt` | Trivial | Usability |
