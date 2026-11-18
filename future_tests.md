# Future tests: once ULTRASAT parameters are known

Tests to run when ULTRASAT parameters arrive. Every number they depend on
lives in [`mission_parameters/ultrasat.json`](mission_parameters/ultrasat.json),
not in this file or in the code, so each run can name the parameter version
it used.

## How to use this file

1. When a new ULTRASAT number arrives, update `mission_parameters/ultrasat.json`:
   set `value`, change `status` (`TBD` → `nominal`/`measured`), fill in
   `source`, bump `version` (MAJOR.MINOR.PATCH), add a `changelog` entry, and
   commit. The loader refuses a version with no changelog entry, and a null
   value whose status isn't `TBD`.
2. Find the tests below that list that parameter under **Needs**. Run them
   and record the result in the **Results log** at the bottom, with the
   parameter file version and the git commit.
3. Bump the version as follows:
   * PATCH: a source or wording fix, no value changed.
   * MINOR: a value is added or refined.
   * MAJOR: the schema changes, or a value is replaced by a different measurement.

Check what's still missing:

```bash
python -c "import oh_h2o; mp = oh_h2o.load_mission_parameters('mission_parameters/ultrasat.json'); \
print(mp.label, [n for n, p in mp.parameters.items() if p['status'] == 'TBD'])"
```

---

## A. OH chemistry and calibration (oh_h2o)

### A1. Does the ULTRASAT band see OH at all?
- **Needs:** `bandpass`, `throughput_curve`
- **Why:** OH (0-0) emission is near 309 nm. The nominal ULTRASAT band in the
  file (230–290 nm, status `unconfirmed`) ends below that. Before any OH
  production-rate work, find the throughput at 305–312 nm: the red edge,
  the red leak, or a dedicated filter.
- **Test:** convolve the OH fluorescence spectrum (the one swift_comet_pipeline
  uses for UVW1, Bodewits et al. 2019) with `throughput_curve`. Report the
  count rate per unit OH flux.
- **Pass:** a non-zero response, and an expected OH signal-to-noise for 46P-like
  activity (Q ~ 10²⁸ s⁻¹ at 1 au) above 5 in one exposure. If not, the OH
  goal needs another route, and every other test in section A is moot.

### A2. OH flux factor
- **Needs:** `throughput_curve`, `oh_band`
- **Test:** compute `oh_flux_per_count_rate` the way swift_comet_pipeline does
  for UVW1, write it into the file, and check that reproducing the Swift UVW1
  number from the Swift effective area gives 1.275 × 10⁻¹² within 2%.

### A3. Continuum scale β and reddening
- **Needs:** `throughput_curve`, `oh_band`, `continuum_band`
- **Test:** β = solar count rate in the OH band / solar count rate in the
  continuum band, using `swift_comet_pipeline.modeling.spectra.solar` with
  ULTRASAT's curve. Then fit `beta_reddening_slope` over 0–30 %/100 nm.
- **Pass:** for Swift, the same code gives the 0.090 in
  `swift_uvot.json`. For ULTRASAT, β and its slope go into the file.
- **Open question:** if ULTRASAT has a single band, the continuum has to come
  from somewhere else (a model or a concurrent ground-based image). Then
  `continuum_band` holds the method, and the model needs a new
  `OHCalibration`.

### A4. Same comet, two instruments
- **Needs:** A1–A3 done, plus a comet seen by ULTRASAT and Swift (or another
  OH instrument) within a few days.
- **Test:** run both hand-off files through `oh_h2o` with each mission file.
- **Pass:** aperture-matched Q(H2O) agrees within the combined errors at
  matching km radii.

### A5. End-to-end on a simulated ULTRASAT frame
- **Needs:** `pixel_scale`, `oh_flux_per_count_rate`, `beta`, `gain`, `read_noise`
- **Test:** inject a vectorial-model OH coma (Q known) plus a dust coma
  scaled by β into a virtual ULTRASAT frame (`comet_testbench.pointing.virtual_frame`
  at `pixel_scale`), run the bench, then `oh_h2o`.
- **Pass:** the Q recovered at the converged radius is within 10% of the
  injected Q.

---

## B. Plate scale and photometry (comet_testbench)

These come from the 2026-11-18 plate-scale audit (Swift ~1"/px against
ULTRASAT ~5.4"/px). The audit proposed fixes but didn't make them. Each one
becomes a test once `pixel_scale` (and, where listed, the detector numbers) is
confirmed.

### B1. Plate-scale invariance
- **Needs:** `pixel_scale`
- **Finding:** every existing test runs at 1"/px (`tests/conftest.py`, `SCALE`).
  A synthetic 1/ρ coma rendered at 5.4"/px recovered **50%** of the true flux in
  a 3" aperture (0.56 px), against 98% at 1"/px. Apertures of 12" and up agree
  within 1%.
- **Test:** run the offline suite at both scales: the same coma and the same
  radii in arcsec/km.
- **Pass:** enclosed rate agrees within 2% for radii ≥ 2 px. Below that,
  results are flagged (B4).

### B2. Recentring at coarse pixels
- **Needs:** `pixel_scale`
- **Finding:** a 10" recentre box gave ~1.3" centroid error at 5.4"/px, against
  0.1" at 1"/px.
- **Test:** inject comae at sub-pixel offsets and measure the centroid error
  against box size.
- **Pass:** error < 0.25 px. Set the default `recenter_box_arcsec` in
  pixels-per-scale terms if needed.

### B3. Local pixel scale and distortion
- **Needs:** flight WCS (distortion terms) from the first images
- **Finding:** `pixel_scale_arcsec` is one mean value at CRPIX, with no local
  scale, no distortion and no skew check. Over a 20° × 10° field that matters
  more than over UVOT's 17'.
- **Test:** compare the local scale (from the WCS Jacobian at the comet)
  against the CRPIX value across the field. Convert apertures with the local
  scale.
- **Pass:** radius_km is right to 0.5% anywhere in the field.

### B4. Minimum aperture size
- **Needs:** `pixel_scale`
- **Finding:** there's no check on apertures smaller than a pixel or two.
- **Test / pass:** apertures under 2 px radius carry a flag in the hand-off,
  and `oh_h2o` skips them by default.

### B5. Frame validation
- **Needs:** first real or simulated ULTRASAT FITS headers
- **Finding:** EXPOSURE of NaN or 0 isn't caught, and frames with a bad WCS are
  skipped silently.
- **Test / pass:** each such frame raises or logs a named reason, and
  `comets.csv` lists skipped frames.

### B6. Detector noise
- **Needs:** `gain`, `read_noise`
- **Finding:** errors assume Poisson counts with gain 1 and no read noise.
- **Test:** put gain and read noise into the `UltraSat` adapter, then compare
  aperture errors against the scatter of repeated synthetic realisations.
- **Pass:** reported errors are within 10% of the empirical scatter.

### B7. Defaults in arcsec, not pixels
- **Needs:** `pixel_scale`
- **Finding:** `benchmarks.py` defaults are in pixels (100/50/10/20 px), and
  these mean very different sky areas at 1 and 5.4 "/px.
- **Test / pass:** defaults are given in arcsec, and the benchmark results at
  both scales are comparable.

### B8. Background choice
- **Needs:** `pixel_scale`, `field_of_view`
- **Finding:** the default background is the whole-frame "global" one. For
  comets at 0.1–0.2 au the coma fills the UVOT field, and the background choice
  moves a 20" aperture rate by 10–33%. ULTRASAT's large field changes this:
  there will usually be true sky, but over a varying Galactic background.
- **Test:** run aperture stability (`benchmarks.aperture_stability`) with local
  annulus against global background on simulated ULTRASAT fields at low and
  high Galactic latitude.
- **Pass:** pick the default that keeps the spread under 5%. The 46P
  Swift run (40,000 km aperture reaching past the background annulus) is the
  reference failure case.

### B9. Aperture coverage
- **Needs:** `pixel_scale`, `field_of_view`
- **Test / pass:** apertures with coverage < 0.95 (detector edge, chip gaps)
  carry a flag that `oh_h2o` honours.

---

## C. Observing geometry

### C1. Spacecraft ephemeris
- **Needs:** `horizons_id`
- **Test:** set `UltraSat(horizons_id=...)`, then rerun `ephemeris_offset` on
  the first frames.
- **Pass:** median offset < 1 px. Until then, the observer is the
  simulator's fixed site (`observer_site`), which is only an approximation for
  a geostationary spacecraft.

---

## Results log

| Date | Test | Parameter file version | Commit | Result |
|---|---|---|---|---|
| | | | | |
