from typing import Tuple, List
from numpy import pi

"""Common functions used in the implementation of acoustic absorbers.

Standard functions for calculating acoustical parameters will be provided
in this module. 

Typical usage example:
  Zc, kc = delany_basley(frequency_vector, sigma=4000, c0=343, rho0=1.21)
"""


def delany_bazley(
    f: List[float], sigma: float, c0: float, rho0: float
) -> Tuple[List[float], List[float]]:
    """Calulates the characteristic impedance and characteristic wave number of a material using the Delany-Bazley model.

        The analysis frequency range should be limited by  0.01 < (f/sigma) < 1.00.
    Args:
      f:
        A vector with all the frequency values.
      sigma:
        Flux resistivity of the material, in [N*s/m^4].
      c0:
        Speed of sound in air, in [m/s].
      rho0:
        Specific mass of air, in [kg/m^3]

    Returns:
      A tuple with two elements, the first one being a vector of specific impedances and
      the second a vector of wave numbers. These vectors will have the same length as `f`.
    """
    # Characteristic impedance
    Zc = (rho0 * c0) * (
        (1 + (9.08 * ((1e3 * (f / sigma)) ** (-0.75))))
        - (1j * 11.9 * ((1e3 * (f / sigma)) ** (-0.73)))
    )  
    
    # Characteristic wave number
    kc = ((2 * pi * f) / c0) * (
        (10.3 * ((1e3 * (f / sigma)) ** (-0.59)))
        + (1j * (1 + (10.8 * ((1e3 * (f / sigma)) ** (-0.7)))))
    )  

    return (Zc, kc)
