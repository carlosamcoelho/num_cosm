import numpy as np

def acceptanceRate(accepted):
    """
    Calculates the fraction of MCMC proposals that were effectively accepted
    during the Markov-Chain run. 

    Parameters
    ----------
    accepted : array_like
        Boolean array with shape ``(nSteps,)`` with the acceptance of the
        proposed value at a given step of the chain.

    Returns
    -------
    accRate : float
        Number between 0 and 1 for the acceptance rate of MCMC proposals.

    """
    accepted = np.asarray(accepted, dtype=bool)
    accRate = np.mean(accepted)
    return accRate

def autoCorrelation(chain, maxLag):
    """
    Estimates the autocorrelation function of each parameter in a chain.

    Parameters
    ----------
    chain : array_like
        MCMC samples with shape ``(nSteps, nDim)``. Warm-up samples
        should be removed before calling this function.
    maxLag : int
        Maximum lag included in the autocorrelation estimate.

    Returns
    -------
    acf : ndarray
        Autocorrelation coefficients with shape
        ``(maxLag + 1, nDim)``. The first row is equal to one.

    Notes
    -----
    Independent chains should be analysed separately rather than
    concatenated before computing the autocorrelation function.
    """
    
    chain = np.asarray(chain, dtype=float)

    # Checking validity of introduced parameters
    if chain.ndim != 2:
        raise ValueError("chain must have shape (nSteps, nDim)!")
    nSteps, nDim = chain.shape
    
    if not isinstance(maxLag, (int, np.integer)):
        raise TypeError("maxLag must be an integer!")
    if maxLag < 0 or maxLag >= nSteps:
        raise ValueError("maxLag must satisfy 0 <= maxLag < nSteps!")

    centered = chain - np.mean(chain, axis=0)
    variance = np.mean(centered**2, axis=0)

    if np.any(variance == 0):
        raise ValueError("Autocorrelation is undefined for a constant parameter!")

    acf = np.empty((maxLag + 1, nDim))
    acf[0] = 1.0

    for lag in range(1, maxLag + 1):
        covariance = np.mean(centered[:-lag] * centered[lag:],axis=0)
        acf[lag] = covariance / variance
    return acf

def effectiveSampleSize(chain, maxLag):
    """
    Calculates de effectiveness of the number of MCMC steps by comparing
    the self-evolution of the Markov-Chain sampled.

    Parameters
    ----------
    chain : array_like
        MCMC samples with shape ``(nSteps, nDim)``. Warm-up samples
        should be removed before calling this function.
    maxLag : int
        Maximum lag included in the autocorrelation estimate.

    Returns
    -------
    nEff : ndarray
        Effective number of MCMC steps for each parameter. Is an array
        with shape ``(nDim,)``.

    Notes
    -----
    Uses the sum of the autoCorrelation at different lag values to
    estimate the effective sample size.
    """
    chain = np.asarray(chain, dtype=float)

    # Checking validity of introduced parameters
    if chain.ndim != 2:
        raise ValueError("chain must have shape (nSteps, nDim)!")
    if not isinstance(maxLag, (int, np.integer)):
        raise TypeError("maxLag must be an integer!")
    nSteps, nDim = chain.shape
    
    if maxLag < 0 or maxLag >= nSteps:
        raise ValueError("maxLag must satisfy 0 <= maxLag < nSteps!")
    acf = autoCorrelation(chain, maxLag)

    tInt = 1 + 2*np.sum(acf, axis=0)
    nEff = nSteps / tInt
    return nEff

def splitRhat(chains):
    """
    Computes the split-Rhat convergence diagnostic.

    Parameters
    ----------
    chains : array_like
        Independent post-warm-up chains with shape
        ``(nChain, nSteps, nDim)``.

    Returns
    -------
    rHat : ndarray
        Split-Rhat value for each parameter, with shape ``(nDim,)``.

    Notes
    -----
    Each chain is divided into two halves before comparing the
    within-chain and between-chain variances. Values close to unity
    indicate consistency between chains and between the first and
    second halves of each chain.
    """
    
    chains = np.asarray(chains, dtype=float)

    if chains.ndim != 3:
        raise ValueError("chains must have shape (nChain, nSteps, nDim)!")

    nChain, nSteps, nDim = chains.shape

    if nChain < 2:
        raise ValueError("At least two independent chains are required!")

    nHalf = nSteps // 2

    if nHalf < 2:
        raise ValueError("Each split chain must contain at least two samples!")

    # If nSteps is odd, discard the central sample
    firstHalf  = chains[:, :nHalf, :]
    secondHalf = chains[:, -nHalf:, :]

    splitChains = np.concatenate((firstHalf, secondHalf),axis=0)

    # Shape: (2*nChain, nDim)
    chainMeans = np.mean(splitChains, axis=1)

    # Variance inside each split chain
    chainVars = np.var(splitChains, axis=1, ddof=1)

    # Mean within-chain variance
    W = np.mean(chainVars, axis=0)
    if np.any(W <= 0):
        raise ValueError("Rhat is undefined for parameters with zero within-chain variance!")

    # Between-chain variance
    B = nHalf * np.var(chainMeans, axis=0, ddof=1)

    varHat = ((nHalf - 1) / nHalf * W + B / nHalf)
    rHat = np.sqrt(varHat / W)

    return rHat
