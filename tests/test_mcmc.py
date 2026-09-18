import numpy as np
import numcosmo_sne.mcmc as mcmc
import pytest

# logPDFs ----------------------------------------------------------------------------------------------------------------------

# Regular 1D gaussian logPDF (differing by a normalization constant)
def log_gaussian_1d(theta, mu=2.0, sigma=1.5):
    return -0.5 * ((theta[0] - mu) / sigma)**2

# Correlated 2D gaussian logPDF
def log_gaussian_2d(theta, mu=np.array([2.0, -1.0]), Cov=np.array([[2.25, 0.9],[0.9,  1.0]])):
    delta = theta - mu
    return -0.5 * delta @ np.linalg.solve(Cov, delta)

# logPDF with probability only in theta[0]=0.0
def point_support(theta):
    if theta[0] == 0.0:
        return 0.0
    return -np.inf

# TESTS ----------------------------------------------------------------------------------------------------------------------

# Tests the shapes of the output arrays from metropolisHastings.
def test_output_shapes(nSteps=100, seed=12345):
    initPos = np.array([0.0, 0.0])
    Cov = np.eye(2)

    chain, accepted, logp = metropolisHastings(log_gaussian_2d, initPos, nSteps, Cov, seed=seed)

    assert chain.shape == (nSteps, 2), "Incorrect shape of 'chain'!"
    assert accepted.shape == (nSteps - 1,), "Incorrect shape of 'accepted'!"
    assert logp.shape == (nSteps,), "Incorrect shape of 'lnProbChain'!"

# Tests if the MCMC generates the same walk for a given seed across different executions.
def test_reproducibility(seed=12345):
    args = (log_gaussian_1d, np.array([0.0]), 1000, np.array([[4.0]]))

    chain1, accepted1, logp1 = metropolisHastings(*args, seed=seed)
    chain2, accepted2, logp2 = metropolisHastings(*args, seed=seed)

    np.testing.assert_array_equal(chain1, chain2), "Incosistent 'chain' across runs!"
    np.testing.assert_array_equal(accepted1, accepted2), "Incosistent 'accepted' across runs!"
    np.testing.assert_array_equal(logp1, logp2), "Incosistent 'lnProbChain' across runs!"

# Tests if rejections repeat the previous state in the random walk using a point_suport logPDF.
def test_rejected_proposals_repeat_state(nSteps = 100, seed=12345):
    chain, accepted, logp = metropolisHastings(point_support, np.array([0.0]), nSteps, np.array([[1.0]]), seed=seed)

    assert np.all(chain == 0.0), "A value with probability -inf was accepted in 'chain'!"
    assert not np.any(accepted), "A value with probability -inf was accepted in 'accepted'!"
    assert np.all(logp == 0.0), "The point_support test must always give 'lnProbChain' of 0.0 (probability=100%)!"

# Compares the log probability given as output with the calculation from the function applied to the chain values.
def test_logprob_matches_chain(seed=12345):
    chain, accepted, logp = metropolisHastings(log_gaussian_1d, np.array([0.0]), 1000, np.array([[4.0]]), seed=seed)

    expected = np.array([log_gaussian_1d(theta) for theta in chain])
    assert np.allclose(logp, expected), "Value of 'lnProbChain' is different from 'lnProb(chain)'!"

# Tests if the MCMC recovers the mean and std of a simple gaussian distribution.
def test_recovers_1d_gaussian(nSteps = 30000, mu = 2.0, sigma = 1.5, accMax=0.9, accMin=0.2):
    muTrue = mu
    covTrue = cov
    
    chain, accepted, logp = metropolisHastings(log_gaussian_1d, np.array([0.0]), nSteps, np.array([[4.0]]), seed=12345)
    samples = chain[nSteps // 10:, 0]

    muMCMC = np.mean(samples)
    sigmaMCMC = np.std(samples, ddof=1)

    assert np.isclose(muMCMC, muTrue, atol=0.05), f"1D gaussian test: Mean {muMCMC} is different from expected {muTrue} within an absolute tolerance of 0.05 for nSteps={nSteps}."
    assert np.isclose(sigmaMCMC, sigmaTrue, atol=0.05), f"1D gaussian test: Std {sigmaMCMC} is different from expected {sigmaMCMC} within an absolute tolerance of 0.05 for nSteps={nSteps}."

    acceptanceRate = np.mean(accepted)
    assert acceptanceRate > accMin, f"1D gaussian test: Acceptance rate of {acceptanceRate} is too low! (Under {accMin})"
    assert acceptanceRate < accMax, f"1D gaussian test: Acceptance rate of {acceptanceRate} is too high! (Over {accMax})"

# Tests the MCMC for a bidimensional correlated gaussian logPDF (uses an isotropic "eye" walk).
def test_recovers_correlated_2d_gaussian(nSteps = 60000, mu=[2.0, -1.0], cov=[[2.25, 0.9], [0.9,  1.0]], accMax=0.9, accMin=0.2):
    muTrue = np.array(mu)
    covTrue = np.array(cov)

    chain, accepted, logp = metropolisHastings(log_gaussian_2d, np.array([0.0, 0.0]), nSteps, np.eye(2), seed=12345)
    samples = chain[nSteps // 10:]

    muMCMC = np.mean(samples, axis=0)
    covMCMC = np.cov(samples.T)

    assert np.allclose(muMCMC, muTrue, atol=0.05), f"2D gaussian test: Mean {muMCMC} is different from expected {muTrue} within an absolute tolerance of 0.05 for nSteps={nSteps}."
    assert np.allclose(covMCMC, covTrue, rtol=0.06, atol=0.05), f"2D gaussian test: Covariance \n{covMCMC}\n is different from expected \n{covTrue}\n within an absolute tolerance of 0.05 for nSteps={nSteps}."

    acceptanceRate = np.mean(accepted)
    assert acceptanceRate > accMin, f"2D gaussian test: Acceptance rate of {acceptanceRate} is too low! (Under {accMin})"
    assert acceptanceRate < accMax, f"2D gaussian test: Acceptance rate of {acceptanceRate} is too high! (Over {accMax})"
