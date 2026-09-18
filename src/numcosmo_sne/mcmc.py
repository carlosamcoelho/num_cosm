import numpy as np
import inspect

# --- func: metropolisHastings ----------------------------------------------
def metropolisHastings(lnProb, initPos, nSteps, Cov, seed=None):
    """
    Samples a target log-probability distribution using a Gaussian
    random-walk Metropolis-Hastings algorithm.

    Parameters
    ----------
    lnProb : callable
        Function evaluated as ``lnProb(theta)``, where ``theta`` is a
        one-dimensional parameter vector (for example, initPos). 
        It must return a scalar log-probability. 
        ``-np.inf`` may be used for points outside the allowed parameter
        space, since they are always rejected. Must not return ``+np.inf``
        or NaN.
    initPos : array_like
        Initial parameter vector with shape ``(nDim,)``. Its
        log-probability must be finite.
    nSteps : int
        Total number of states stored in the Markov chain, including
        the initial state.
    Cov : array_like
        Positive-definite covariance matrix of the Gaussian proposal,
        with shape ``(nDim, nDim)``.
    seed : int, optional
        Seed used to initialize NumPy's random number generator.

    Returns
    -------
    chain : ndarray
        Markov chain with shape ``(nSteps, nDim)``.
    accepted : ndarray
        Boolean array with shape ``(nSteps - 1,)`` indicating whether
        each proposal was accepted.
    lnProbChain : ndarray
        Log-probability associated with every stored state, with shape
        ``(nSteps,)``.

    Raises
    ------
    ValueError
        If the initial position, proposal covariance, number of steps,
        or initial log-probability is invalid.
    TypeError
        If ``lnProb`` is not callable with a single parameter vector
        or does not return a scalar.

    Notes
    -----
    The proposal distribution is symmetric,

    ``theta' = theta + L xi``,

    where ``L`` is the Cholesky factor of ``Cov`` and ``xi ~ N(0, I)``.
    Rejected proposals repeat the current state in the chain.
    """
    
    rng = np.random.default_rng(seed)
    initPos = np.asarray(initPos, dtype=float)
    Cov = np.asarray(Cov, dtype=float)
    nDim = initPos.size

    # Verifying dimensions and initial values
    if initPos.ndim != 1:
        raise ValueError("initPos must be an one-dimensional array!")
    if Cov.shape != (nDim, nDim):
        raise ValueError("Cov must be a square matrix with dimensions of the number of parameters!")
    if not np.all(np.isfinite(Cov)):
        raise ValueError("Cov must contain only finite values!")
    if not np.allclose(Cov, Cov.T):
        raise ValueError("Cov must be symmetric!")
    if not isinstance(nSteps, (int, np.integer)):
        raise TypeError("nSteps must be an integer!")
    if nSteps < 1:
        raise ValueError("nSteps must be at least 1!")
    if not np.all(np.isfinite(initPos)):
        raise ValueError("initPos must contain only finite values!")

    # Verifying if lnProb is a function
    if not callable(lnProb):
        raise TypeError("lnProb must be callable!")
    # Verifying if lnProb can be called with initPos as its only required argument
    try:
        inspect.signature(lnProb).bind(initPos)
    except TypeError as exc:
        raise TypeError("lnProb must be callable as lnProb(initPos), with no additional required arguments.") from exc

    # If lnProb is a valid function, the probability of initPos can be calculated
    lnProbInitial = lnProb(initPos)

    # Checking the validity of initPos probability
    if not np.isscalar(lnProbInitial):
        raise TypeError("lnProb must return a scalar!")
    if not np.isfinite(lnProbInitial):
        raise ValueError("initPos must have a finite log-probability!")

    # Applying the Cholesky decomposition of Cov
    try:
        L = np.linalg.cholesky(Cov)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Cov must be positive definite!") from exc

    # Creating the necessary arrays for MCMC
    chain = np.zeros((nSteps, nDim))
    accepted = np.zeros(nSteps-1, dtype=bool)
    lnProbChain = np.zeros(nSteps, dtype=float)

    # Generating all necessary random numbers before loop
    xi = rng.normal(size=(nSteps - 1, nDim))
    u  = rng.random(nSteps - 1)

    # MCMC random walk ---------------------------------------------------------------
    chain[0]       = initPos # Initial parameters
    lnProbChain[0] = lnProbInitial # Assigning the value after the necessary guards
    
    for i in range(nSteps - 1):
        current = chain[i]
        proposal = current + L @ xi[i]

        lnProbProposal = lnProb(proposal)

        # Checking if there are invalid values for probability
        if not np.isscalar(lnProbProposal):
            print("proposal = ", proposal)
            raise TypeError(f"lnProb must return a scalar! Failed at MCMC step {i}.")
        if np.isnan(lnProbProposal) or np.isposinf(lnProbProposal):
            print("proposal = ", proposal)
            raise ValueError(f"MCMC at step {i} failed due to NaN or +infinity lnProb!")

        # Metropolis acceptance test based on lnProb
        if np.log(u[i]) < min(0.0, lnProbProposal - lnProbChain[i]):
            chain      [i+1] = proposal 
            accepted   [i]   = True
            lnProbChain[i+1] = lnProbProposal
        else:
            chain      [i+1] = current
            accepted   [i]   = False
            lnProbChain[i+1] = lnProbChain[i]

    return chain, accepted, lnProbChain