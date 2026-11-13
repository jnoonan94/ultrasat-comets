"""Comet test bench for the UltraSat OH pipeline.

Instrument-agnostic tools to (a) find comets in an image from its WCS,
(b) measure coma flux in fixed and iterative apertures, and (c) hand the
photometry and geometry to the separately developed OH/H2O package as a
plain-text ECSV file.
Swift/UVOT is the first supported public dataset.
"""

from .image import ImageFrame, iter_frames, load_frames
from .ephemeris import (ObservingGeometry, SbIdentRecord, normalize_designation, query_horizons,
                        sb_ident_comets, sb_ident_query)
from .instruments import InstrumentAdapter, SwiftUVOT, UltraSat, adapter_for
from .field_search import CometInField, find_comets
from .photometry import (ApertureResult, Background, GrowthCurve, estimate_background,
                         fixed_aperture_photometry, iterative_aperture_photometry, recenter)
from .oh import SCHEMA as HANDOFF_SCHEMA, handoff_table, read_handoff, write_handoff
from .pointing import PointingConstraints, search_pointings, virtual_frame
from .pipeline import BenchConfig, measure_frame, run, write_results

__all__ = [n for n in dir() if not n.startswith("_")]
