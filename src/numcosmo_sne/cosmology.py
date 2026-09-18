import numpy as np
import scipy as sp
from scipy import constants as cte

class NonPhysicalCosmologyError(ValueError):
    """Raised when a cosmological model gives a non-physical FLRW background."""
    pass


def cosmologicalExpansion(z, omega_m, omega_lambda, omega_r=0):
    """
    Calculates the expansion rate normalized by the Hubble constant
    for a given choice of cosmological parameters, at given redshift.

    Parameters
    ----------
    z : float or array-like
        Redshift value (or array of values) where the expansion will be calculated.
    omega_m : float
        Cosmological density for matter (baryonic+DM).
    omega_lambda: float
        Cosmological density for Lambda (assumed as a cosmological constant term).
    omega_r : float, optional
        Cosmological density for radiation.

    Returns
    -------
    E : ndarray
        Expansion rate (for the given z) of this universe.

    Raises
    ------
    NonPhysicalCosmologyError
        If given densities do not correspond to a physically valid universe solution.

    """
    z = np.asarray(z, dtype=float)
    if np.any(z < 0): raise ValueError("Redshift must be non-negative.")

    omega_k = 1 - omega_m - omega_lambda - omega_r

    E2 = (omega_r * (1 + z)**4
        + omega_m * (1 + z)**3
        + omega_k * (1 + z)**2
        + omega_lambda)

    invalid = (~np.isfinite(E2)) | (E2 <= 0)

    if np.any(invalid):
        raise NonPhysicalCosmologyError(
            f"Invalid expansion history for "
            f"omega_m={omega_m:.3f}, omega_lambda={omega_lambda:.3f}, "
            f"omega_r={omega_r:.3f}.")

    E = np.sqrt(E2)

    return E

# --- func: cosmologicalDistances ----------------------------------------------
def cosmologicalDistances(z, H0, omega_m, omega_lambda, omega_r=0, n_grid=257):
    """
    Calculates the different cosmological distances.

    Parameters
    ----------
    z : float or array-like
        Redshift value (or array of values) where the expansion will be calculated.
    H0 : float
        Hubble constant value.
    omega_m : float
        Cosmological density for matter (baryonic+DM).
    omega_lambda: float
        Cosmological density for Lambda (assumed as a cosmological constant term).
    omega_r : float, optional
        Cosmological density for radiation.

    Returns
    -------
    DC : ndarray
        Line-of-sight comoving distance.
    DM : ndarray
        Transverse comoving distance.
    DA : ndarray
        Angular size distance.
    DL : ndarray
        Luminosity distance.

    Raises
    ------
    NonPhysicalCosmologyError
        If given densities do not correspond to a physically valid universe solution.
    """
    z = np.asarray(z, dtype=float)
    if np.any(z < 0): raise ValueError("Redshift must be non-negative.")
    if H0 <= 0:       raise ValueError("H0 must be positive.")
    
    DH = 1e-3*cte.c / H0
    omega_k = 1 - omega_m - omega_lambda - omega_r

    # Determinação da influência de curvatura cosmológica na distância
    if np.isclose(omega_k, 0.0):
        Sk = lambda DC: DC
    elif omega_k > 0:
        Sk = lambda DC: DH / np.sqrt(omega_k) * np.sinh(DC/DH * np.sqrt(omega_k))
    elif omega_k < 0:
        Sk = lambda DC: DH / np.sqrt(-omega_k) * np.sin(DC/DH * np.sqrt(-omega_k))
        
    # Cálculo único de z individual
    if z.size == 1:
        z_scalar = float(z.ravel()[0])
        integDC = lambda z_: 1. / cosmologicalExpansion(z_, omega_m, omega_lambda, omega_r=omega_r)
        DC = DH * sp.integrate.quad(integDC, 0, z_scalar)[0]

    # Cálculo a partir de um vetor z. Calcula-se a partir de interpolação de um z_grid.
    else:
        z_grid = np.linspace(0.0, np.max(z), n_grid)
        E_grid = cosmologicalExpansion(z_grid, omega_m, omega_lambda, omega_r=omega_r)
        
        integDC_grid = 1. / E_grid

        DC_grid = DH * sp.integrate.cumulative_simpson(integDC_grid, x=z_grid, initial=0.0)
        DC = np.interp(z.ravel(), z_grid, DC_grid).reshape(z.shape)
            
    DM = Sk(DC)
    DA = DM/(1+z)
    DL = DM*(1+z)
    return DC, DM, DA, DL

# --- func: cosmologicalTime ----------------------------------------------
def cosmologicalTime(z, H0, omega_m, omega_lambda, omega_r=0, n_grid=257):
    """
    Calculates cosmological timescales.
    
    Parameters
    ----------
    z : float or array-like
        Redshift value (or array of values) where the expansion will be calculated.
    H0 : float
        Hubble constant value.
    omega_m : float
        Cosmological density for matter (baryonic+DM).
    omega_lambda: float
        Cosmological density for Lambda (assumed as a cosmological constant term).
    omega_r : float, optional
        Cosmological density for radiation.

    Returns
    -------
    T : ndarray
        Cosmological time since the Big Bang.
    Eta : ndarray
        Conformal time for light propagation.

    Raises
    ------
    NonPhysicalCosmologyError
        If given densities do not correspond to a physically valid universe solution.

    """
    
    z = np.asarray(z, dtype=float)
    if np.any(z < 0): raise ValueError("Redshift must be non-negative.")
    if H0 <= 0:       raise ValueError("H0 must be positive.")
        
    H0_Gyr = 1e+9*cte.year * H0 / (1e+3 * cte.parsec)

    integEta = lambda z_: 1. / cosmologicalExpansion(z_, omega_m, omega_lambda, omega_r=omega_r)
    integT = lambda z_: integEta(z_) / (1+z_)

    # Cálculo único de z individual
    if z.size == 1:
        z_scalar = float(z.ravel()[0])
        Eta = sp.integrate.quad(integEta, z_scalar, np.inf)[0] / H0_Gyr
        T = sp.integrate.quad(integT, z_scalar, np.inf)[0] / H0_Gyr
        
    else:
        z_grid = np.linspace(0.0, np.max(z), n_grid)
        E_grid = cosmologicalExpansion(z_grid, omega_m, omega_lambda, omega_r=omega_r)
        
        integEta_grid = 1. / E_grid
        integT_grid   = integEta_grid / (1+z_grid)
        
        Eta_from_0 = sp.integrate.cumulative_simpson(integEta_grid, x=z_grid, initial=0.0)
        T_from_0 = sp.integrate.cumulative_simpson(integT_grid, x=z_grid, initial=0.0)

        Eta0 = sp.integrate.quad(integEta, 0.0, np.inf)[0]
        T0 = sp.integrate.quad(integT, 0.0, np.inf)[0]

        Eta_grid = Eta0 - Eta_from_0
        T_grid   = T0   - T_from_0

        Eta = np.interp(z.ravel(), z_grid, Eta_grid).reshape(z.shape) / H0_Gyr
        T   = np.interp(z.ravel(), z_grid, T_grid).reshape(z.shape) / H0_Gyr

    return T, Eta

# --- func: FLRW ----------------------------------------------
def FLRW(z, omega_m, omega_lambda, H0=70, omega_r=0):
    """
    Calculates multiple cosmological quantities at a given redshift.

    Parameters
    ----------
    z : float or array-like
        Redshift value (or array of values) where the expansion will be calculated.
    omega_m : float
        Cosmological density for matter (baryonic+DM).
    omega_lambda: float
        Cosmological density for Lambda (assumed as a cosmological constant term).
    H0 : float, optional
        Hubble constant value.
    omega_r : float, optional
        Cosmological density for radiation.

    Returns
    -------
    E : ndarray
        Expansion rate of this universe. (H/H0)
    H : ndarray
        Hubble parameter at given redshift.
    T : ndarray
        Cosmological time since the Big Bang.
    Eta : ndarray
        Conformal time for light propagation.
    DC : ndarray
        Line-of-sight comoving distance.
    DM : ndarray
        Transverse comoving distance.
    DA : ndarray
        Angular size distance.
    DL : ndarray
        Luminosity distance.
    
    Raises
    ------
    NonPhysicalCosmologyError
        If given densities do not correspond to a physically valid universe solution.

    """
    z = np.asarray(z, dtype=float)
    if np.any(z < 0): raise ValueError("Redshift must be non-negative.")
    if H0 <= 0:       raise ValueError("H0 must be positive.")
        
    E = cosmologicalExpansion(z, omega_m, omega_lambda, omega_r=omega_r)
    H = H0 * E
    
    T, Eta = cosmologicalTime(z, H0, omega_m, omega_lambda, omega_r=omega_r)
    DC, DM, DA, DL = cosmologicalDistances(z, H0, omega_m, omega_lambda, omega_r=omega_r)

    if np.any(~np.isfinite(DL)) or np.any(DL <= 0):
        raise NonPhysicalCosmologyError("Luminosity distance became non-positive.")
    
    Mu = 5*np.log10(DL)+25
    dV = 1e-3*cte.c * DM**2 / H

    return E, H, T, Eta, DC, DM, DA, DL, Mu, dV