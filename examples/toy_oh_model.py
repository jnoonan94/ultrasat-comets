"""A placeholder OH model showing the interface the real OH package implements.

NOT a physical model. Run with:
    python -m comet_testbench run ... --beta B --oh-conversion C \
        --oh-model examples.toy_oh_model:ToyOHModel
"""

import numpy as np


class ToyOHModel:
    name = "toy-linear"
    version = "0"

    def aperture_flux(self, production_rate, radius_km, geometry):
        # erg s-1 cm-2; scales like Q * rho / (r_h^2 * Delta) for illustration only.
        return 1e-41 * production_rate * np.asarray(radius_km) / (geometry.r_h**2 * geometry.delta)
