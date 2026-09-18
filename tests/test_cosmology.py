import numpy as np
import scipy as sp
from scipy import constants as cte
import numcosmo_sne.cosmology as cosmo
import pytest

# Tests if the expansion at z=0 is normalized (i.e., equal to 1).
def test_E0(omega_m = 0.3, omega_lambda = 0.7):
    assert np.isclose(cosmo.cosmologicalExpansion(0, 0.3, 0.7), 1.0), "--- Expansion test failed!"

# Tests the distance equavalence in a flat universe.
def test_flat_DM_equals_DC(omega_m = 0.3):
    z = np.linspace(0.1, 5, 20)
    DC, DM, _, _ = cosmo.cosmologicalDistances(z, 70, omega_m, 1-omega_m)
    assert np.allclose(DM, DC), "--- Flatness test failed!"

# Tests the duality between angular and luminosity distances.
def test_distance_duality(omega_m = 0.3, omega_lambda = 0.7):
    z = np.linspace(0.1, 5, 20)
    _, _, DA, DL = cosmo.cosmologicalDistances(z, 70, 0.3, 0.7)
    assert np.allclose(DL, (1 + z)**2 * DA), "--- Distance duality test failed!"

# Tests the validity of the Hubble law approximation at low redshift.
def test_hubble_law(z=1e-4, H0=70, rtol=1e-3):
    _, _, _, DL = cosmo.cosmologicalDistances(z, H0, omega_m=0.3, omega_lambda=0.7)
    DH = 1e-3 * cte.c / H0
    assert np.isclose(DL / (DH * z), 1.0, rtol=rtol), "--- Hubble law test failed!"

    