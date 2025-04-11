import numpy as np
from typing import Annotated, List, Tuple


"""Standard macroscopic models used to model acoustic absorbers.

Standard functions for calculating acoustical parameters will be provided
in this module. 

Typical usage example:
  Zc, kc = delany_basley(frequency_vector, sigma=4000, c0=343, rho0=1.21)
  Zc, kc = jca(frequency_vector, sigma=4000, tortuosity=1.1, porosity=0.95, viscous_characteristic_length=1e-4)
"""


def delany_bazley(
    f: List[float], sigma: float, c0: float = 343, rho0: float = 1.21
) -> Tuple[List[float], List[float]]:
    """Calulates the characteristic impedance and characteristic wave number of a material using the Delany-Bazley model.

        The analysis frequency range should be limited by  0.01 < (f/sigma) < 1.00.
    Args:
      f:
        A vector with all the frequency values.
      sigma:
        Flux resistivity of the material, in [N*s/m^4].
      c0:
        Speed of sound in air, in [m/s]. Defaults to 343 [m/s].
      rho0:
        Specific mass of air, in [kg/m^3]. Defaults to 1.21 [kg/m^3]

    Returns:
      A tuple with two elements, the first one being a vector of specific impedances and
      the second a vector of wave numbers. These vectors will have the same length as `f`.


    References:
      [1] E. Brandão, Acústica de salas: Projeto e modelagem. São Paulo: Editora Edgard Blucher, 2016.

    """
    # fmt: off
    # Characteristic impedance
    Zc = (rho0 * c0)*(1 + 9.08*(((1e3*f)/sigma)**(-0.75)) - 1j *11.9*(((1e3*f) / sigma)**(-0.73)))
    # Characteristic wave number
    kc = ((2 * np.pi * f) / c0) * (1 + 10.8*(((1e3*f)/sigma)**(-0.70)) - 1j*10.3*(((1e3*f)/sigma)**(-0.59)))
    # fmt: on
    return (Zc, kc)


def jca(
    f: List[float],
    sigma: float,
    tortuosity: float,
    porosity: float,
    viscous_characteristic_length: float,
    rho0: float = 1.21,
    gamma: float = 1.4,
    air_viscosity: float = 1.84e-5,
    prandtl: float = 0.77,
    P0: float = 101325,
) -> Tuple[List[float], List[float]]:
    """Calulates the characteristic impedance and characteristic wave number of a material using the Johnson-Champoux-Allard model.

    Args:
      f:
        A vector with all the frequency values.
      sigma:
        Flux resistivity of the material, in [N*s/m^4].
      tortuosity:
        Tortuosity of the material. Adimensional value.
      porosity:
        Porosity of the material. Adimensional value, that should be in the range [0, 1].
      viscous_characteristic_length:
        Viscous characteristic length of the material, in [µm].
      rho0:
        Specific mass of air, in [kg/m^3]. Defaults to 1.21 [kg/m^3]
      gamma:
        Ratio of specific heats Cp/Cv. Adimensional value. Defaults to 1.4 [-].
      air_viscosity:
        Viscosity of air, in [Pa*s]. Defaults to 1.84e-5 [Pa*s].
      prandtl:
        Prandtl number for air. Adimensional value. Defaults to 0.77.
      P0:
        Absolute atmosferic pressure, in [Pa]. Defaults to 101325 [Pa].
    Returns:
      A tuple with two elements, the first one being a vector of specific impedances and
      the second a vector of wave numbers. These vectors will have the same length as `f`.

    References:
      [1] E. Brandão, Acústica de salas: Projeto e modelagem. São Paulo: Editora Edgard Blucher, 2016.

    """
    omega = 2 * np.pi * f
    # fmt: off
    brackets_sqrt = np.sqrt(1 + ((4j*(tortuosity**2)*air_viscosity*rho0*omega) / ((sigma**2)*(viscous_characteristic_length**2)*(porosity**2))))
    rho_c = (rho0*tortuosity) * (1 + (((sigma*porosity) / (1j*tortuosity*rho0*omega)) * brackets_sqrt))

    denominator_sqrt = np.sqrt(1 + ((4j*(tortuosity**2)*air_viscosity*rho0*prandtl*omega) / ((sigma**2)*(viscous_characteristic_length**2)*(porosity**2))))
    K = gamma*P0 / (gamma - ((gamma - 1) / (1 + (((sigma*porosity) / (1j*tortuosity*rho0*prandtl*omega)) * denominator_sqrt))))
    # fmt: on

    Zc = np.sqrt(K * rho_c)
    kc = omega * np.sqrt(rho_c / K)

    return (Zc, kc)