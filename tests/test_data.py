import numpy as np
import numcosmo_sne.data as data
from pathlib import Path
import pytest

# Tests if the data is consistent with the assumptions of the implementation.
def test_data():
    DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "union2.1"

    names, z, mu, mu_err, p_low, Wmat = data.readUnionData(DATA_DIR)

    assert np.shape(Wmat)[0] == np.shape(Wmat)[0], "Covariance matrix must be a square matrix!"
    assert np.shape(Wmat)[0] == len(z), "Covariance matrix and data have different dimensions!"

    assert np.all(np.isfinite(z)), "Found an invalid value in z!"
    assert np.all(np.isfinite(mu)), "Found an invalid value in mu!"
    assert np.all(np.isfinite(mu_err)), "Found an invalid value in mu_err!"

    assert np.all(mu_err > 0), "All uncertainties mu_err must be positive!"
    assert np.all(z > 0), "All redshift values z must be positive!"
    assert np.all((p_low >= 0) & (p_low <= 1)), "Probability values p_low must be between 0 and 1!"

    assert np.all(np.isfinite(Wmat)), "Found an invalid value in Wmat!"
    assert np.allclose(Wmat, Wmat.T), "The covariance matrix must be symmetrical!"

    null_zero = Wmat @ np.ones(580)
    null_p = Wmat @ p_low

    assert np.allclose(null_zero, 0.0, atol=1e-5), "The covariance matrix is not corrected for zero value!"
    assert np.allclose(null_p, 0.0, atol=1e-5), "The covariance matrix is not corrected for low galaxy mass!"

# Tests if the posterior function for the union data is consistent for the MCMC application
def test_union_log_posterior(omega_m=0.3, omega_lambda=0.7):
    DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "union2.1"

    _, z, mu, _, _, Wmat = data.readUnionData(DATA_DIR)
    lnProb = makeUnionLogPos(z, mu, Wmat)

    theta = np.array([omega_m, omega_lambda])
    value = lnProb(theta)

    assert np.isscalar(value), "The union posterior calculation must return a scalar!"
    assert np.isfinite(value), "The union posterior must be finite!"

    assert np.isneginf(lnProb(np.array([3.0, 0.7]))), "The posterior value outside the inequality prior must be -inf!"
    assert np.isneginf(lnProb(np.array([0, 3]))), "The posterior at a non-physical parameter must be -inf!"

    

    


