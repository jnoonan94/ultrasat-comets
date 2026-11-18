"""OH -> H2O production rates from comet aperture photometry.

Built on swift_comet_pipeline (Oset et al.), imported as a dependency: this
package adds an instrument-agnostic interface and reads the comet test bench
hand-off file. See README.md in this folder.

    import oh_h2o
    oh_h2o.configure(data_dir="~/src/swift_comet_pipeline/data", quality="medium")
    result = oh_h2o.run("results/handoff/46P_photometry.ecsv",
                        oh_h2o.SwiftUVOTCalibration(), dust_redness=[0, 10, 20])
    oh_h2o.write_results(result, "results/oh/46P_water.ecsv")
"""

__version__ = "0.1.0"

from .backend import configure, settings  # noqa: E402
from .calibration import (  # noqa: E402
    FixedCalibration,
    OHCalibration,
    SwiftUVOTCalibration,
    calibration_for,
)
from .mission import (  # noqa: E402
    MissionParameterError,
    MissionParameters,
    calibration_from_mission,
    load_mission_parameters,
)
from .model import (  # noqa: E402
    ContractError,
    continuum_at_radius,
    load_handoff,
    pair_frames,
    run,
    write_results,
)
from .physics import (  # noqa: E402
    Geometry,
    continuum_subtract,
    g_factor_1au,
    oh_flux,
    oh_molecules,
    q_h2o_in_aperture,
    q_h2o_total,
)

__all__ = [
    "configure", "settings", "OHCalibration", "SwiftUVOTCalibration", "FixedCalibration",
    "calibration_for", "ContractError", "load_handoff", "pair_frames", "continuum_at_radius",
    "run", "write_results", "Geometry", "continuum_subtract", "oh_flux", "oh_molecules",
    "q_h2o_in_aperture", "q_h2o_total", "g_factor_1au", "MissionParameters",
    "MissionParameterError", "load_mission_parameters", "calibration_from_mission",
]
