"""Comet test bench for the UltraSat OH pipeline.

Instrument-agnostic tools to (a) find comets in an image from its WCS,
(b) measure coma flux in fixed and iterative apertures, and (c) hand
continuum-subtracted OH fluxes to a separately developed OH model package.
Swift/UVOT is the first supported public dataset.
"""

from .image import ImageFrame, iter_frames, load_frames
from .ephemeris import (ObservingGeometry, SbIdentRecord, normalize_designation, query_horizons,
                        sb_ident_comets, sb_ident_query)
from .instruments import InstrumentAdapter, SwiftUVOT, UltraSat, adapter_for
from .field_search import CometInField, find_comets
from .photometry import (ApertureResult, Background, GrowthCurve, estimate_background,
                         fixed_aperture_photometry, iterative_aperture_photometry, recenter)
from .oh import OHMeasurement, OHModel, ProductionRateFit, fit_production_rate, load_model, subtract_continuum
from .pointing import PointingConstraints, search_pointings, virtual_frame
from .pipeline import BenchConfig, measure_frame, oh_measurements, run, write_results

__all__ = [n for n in dir() if not n.startswith("_")]
