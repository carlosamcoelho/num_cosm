# Numerical Cosmology — Union2.1 Supernova MCMC


This repository contains the implementation developed as an exercise for the
Numerical Cosmology discipline at IF-USP, São Paulo (SP), Brazil. The objective is to 
implement a cosmological parameter inference of Omega_matter and Omega_lambda from the 
Union2.1 Type Ia supernova sample, without imposing a flatness constraint.

## Methods
This project implements:
- FLRW background expansion and cosmological distances;
- Union2.1 data and weight-matrix loading;
- Union2.1 Gaussian log-likelihood;
- Rejection of non-physical cosmologies through tailored exception handlings.
- Gaussian random-walk Metropolis-Hastings sampling;
- MCMC validation using analytic 1D and correlated 2D Gaussian targets;
- Autocorrelation, effective sample size, acceptance fraction and
  split-Rhat diagnostics.

We apply the implementation, using the available .ipynb notebooks, to the Union2.1 data
and recover the 68% and 95% probability contours of the cosmological parameters.

It is important to note that the Union2.1 covariance weight matrix is already 
marginalized over the supernova magnitude zero-point, so it was not necessary to measure
the zero-point as a nuisance parameter during the MCMC runs. Additionally, Union2.1 is 
also marginalized for the probability that a SNe event was hosted in a low-mass galaxy.

## Installation

Create/activate the desired Python environment and install the package in editable mode:

    python -m pip install -e .

For development/testing:

    python -m pip install -e ".[dev]"

## Tests

Run the complete test suite with:

    pytest


The tests include cosmological distance identities, Union2.1 data consistency checks, 
and recovery of known one- and two-dimensional Gaussian target distributions by the 
Metropolis-Hastings sampler.

## Analysis

The notebooks reproduce the main stages of the analysis:

1. FLRW background validation;
2. Union2.1 data inspection;
3. validation of the generic MCMC sampler;
4. Union2.1 posterior sampling and convergence diagnostics.

Generated chains are stored in `outputs/chains/` and figures in `outputs/figures/`.

## Output
The primary result is included in `repo/outputs/figs/unionMCMC.png`, and shows
the joint posterior distribution: 

P(\Omega_m,\Omega_\Lambda | Union2.1),

including the desired 68% and 95% credible regions and a comparision with the flatness
condition, among other diagnostics. The `outputs` folder includes some MCMC runs with 
selected seeds, and the diagnostic figures from the given notebooks.
