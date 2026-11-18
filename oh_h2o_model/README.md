# oh_h2o: OH → H2O production rates

The second half of the comet pipeline. It reads the comet test bench hand-off
file (aperture photometry plus observing geometry, schema
`ultrasat-comet-photometry/1`) and returns OH fluxes and water production
rates for every aperture.

The science comes from [swift_comet_pipeline](https://pypi.org/project/swift_comet_pipeline/)
(S. Oset, GPL-3.0), imported as a dependency and wrapped rather than copied.
This package adds three things:

* an instrument-agnostic calibration interface, so ULTRASAT can be added
  without touching the physics;
* reading the test bench hand-off, pairing OH-band and continuum images, and
  matching apertures by physical radius;
* the set-up swift_comet_pipeline needs when it is installed from PyPI.

## Steps and where they come from

| Step | Function here | swift_comet_pipeline function | Instrument-specific? |
|---|---|---|---|
| Continuum scale β | `OHCalibration.beta(redness)` | `photometry.dust.beta_parameter` | yes |
| OH signal = OH band − β × continuum | `continuum_subtract` | (same arithmetic as `builders.aperture_water_production_calculator`) | no |
| OH flux (erg cm⁻² s⁻¹) | `oh_flux` | `modeling.water_production.flux_OH.oh_count_rates_to_flux_factor` | yes (the factor) |
| Number of OH molecules | `oh_molecules` | `modeling.water_production.fluorescence_OH.oh_flux_to_num_oh` (g-factor: Schleicher & A'Hearn 1988) | no |
| Q(H2O) matched to the aperture | `q_h2o_in_aperture` | `modeling.water_production.num_OH_to_Q.num_oh_within_r_to_q_h2o_vectorial` | no |
| Q(H2O) if the aperture held the whole coma | `q_h2o_total` | `num_oh_to_q_h2o_vectorial` | no |

Only two numbers depend on the telescope: β and the OH flux per count rate.
They sit behind `OHCalibration`:

* `SwiftUVOTCalibration(oh_band="UVW1", continuum_band="V")` gets them from
  swift_comet_pipeline (UVW1, UVW2, UVM2 and U have flux factors there).
* `FixedCalibration(name, oh_band, continuum_band, beta_value, flux_per_count_rate, instruments=(...))`
  takes them as numbers (β can be a function of redness). This is the slot for
  ULTRASAT until it has its own class.

`q_h2o` (aperture-matched) is the one to use. `q_h2o_total` is what
swift_comet_pipeline calls the "equivalent" production rate; it is only right
when the aperture holds all the OH, which test bench apertures never do.

## Install

swift_comet_pipeline 0.3 needs **Python 3.11 or 3.12**.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e oh_h2o_model            # pulls swift_comet_pipeline and synphot
pip install -e .                       # the test bench, if you want both in one env
git clone --depth 1 https://github.com/sjoset/swift_comet_pipeline ~/src/swift_comet_pipeline
export SWIFT_COMET_PIPELINE_DATA=~/src/swift_comet_pipeline/data
```

The PyPI wheel of swift_comet_pipeline does not include its `data/` folder
(OH g-factor table, UVOT effective areas). Point `SWIFT_COMET_PIPELINE_DATA`
or `configure(data_dir=...)` at the folder in a clone. It also needs `synphot`
without declaring it; this package's dependencies add it.

## Use

```bash
python -m comet_testbench run data/swift/*/*_sk.img.gz --config examples/swift_46p_config.json
python -m oh_h2o results/handoff/46P_photometry.ecsv --redness 0 10 20 --quality medium
# -> results/oh/46P_water.ecsv
```

```python
import oh_h2o

oh_h2o.configure(quality="medium")          # data dir from $SWIFT_COMET_PIPELINE_DATA
res = oh_h2o.run("results/handoff/46P_photometry.ecsv",
                 oh_h2o.SwiftUVOTCalibration(), dust_redness=[0, 10, 20])
oh_h2o.write_results(res, "results/oh/46P_water.ecsv")
```

`run` also accepts the table from `comet_testbench.oh.handoff_table(...)`
directly, without a file in between.

Output columns (schema `ultrasat-oh-h2o/1`): the image pair and time gap, the
aperture (kind, step, radius_km, converged flag), dust redness, β, OH-band and
continuum rates, OH rate, OH flux, number of OH, `q_h2o`, `q_h2o_total`, all
with 1σ errors, and r_h, ṙ, Δ.

### Pairing and aperture matching

Each OH-band image is paired with the nearest continuum image in time (default
within 60 minutes, `max_dt_minutes`). The continuum rate at each OH aperture's
radius in km is linearly interpolated from the continuum image's apertures
(fixed and growth curve together); radii outside that range are skipped, not
extrapolated.

### Vectorial model

`configure(quality=..., backend=..., cache_dir=...)`. Quality `low` takes
about half a second per heliocentric distance, `medium` a few seconds; results
are cached in `~/.cache/oh_h2o/cache/vectorial_model_cache.sqlite3`. The `rust`
backend needs pyvectorial_au's compiled binary.

## Known swift_comet_pipeline quirks this package works around

* Data files missing from the wheel (above).
* Undeclared `synphot` dependency.
* `num_oh_within_r_to_q_h2o_vectorial` divides by N, N−σ and N+σ, so it fails
  when the OH signal is exactly 0 or equal to its error. Q is linear in N, so
  those cases fall back to scaling its Q for N = 1, with the same error
  convention (σ_Q = √2 σ_N dQ/dN).
* Its state (data paths, vectorial cache path, grid quality, backend) is
  module-global; the vectorial cache path can only be set once per process.

## Tests

```bash
cd oh_h2o_model
SWIFT_COMET_PIPELINE_DATA=~/src/swift_comet_pipeline/data python -m pytest
```

Without the data folder the tests that need it are skipped. One test runs
synthetic UVW1 + V images through the test bench and then through this package.
