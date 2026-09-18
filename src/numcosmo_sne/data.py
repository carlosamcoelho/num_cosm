import numpy as np
import numcosmo_sne.cosmology as cosmo
from pathlib import Path

def readUnionData(data_dir, use_sys=True):
    """
    Reads the union2.1 SNe data txt files available on
    https://supernova.lbl.gov/union/

    Parameters
    ----------
    data_dir : path-like
        Path to the folder that contains the txt data files.
    use_sys : bool, optional
        Changes if the desired ``Wmat`` is sys (``True``)
        or nosys (``False``).

    Returns
    -------
    names : ndarray
        Specified data names for a given SNe.
    z : ndarray
        Measured redshifts of the SNe events.
    mu : ndarray
        Distance modulus of each SNe.
    mu_err : ndarray
        Error in the distance modulus measurement.
    p_lot : ndarray
        Probability that the supernova was hosted by a low-mass galaxy.
    Wmat : ndarray
        Covariance matrix of the data.

    Notes
    -----
    Union data files must have the following names 
    ``sn_z_mu_dmu_plow_union2.1.txt``
    ``sn_wmat_sys_union2.1.txt``
    ``sn_wmat_nosys_union2.1.txt``
    """
    
    sn_file = data_dir / "sn_z_mu_dmu_plow_union2.1.txt"
    if use_sys:
        wmat_file = data_dir / "sn_wmat_sys_union2.1.txt"
    else:
        wmat_file = data_dir / "sn_wmat_nosys_union2.1.txt"

    names = np.loadtxt(sn_file, usecols=0, dtype=np.dtypes.StringDType)
    z, mu, mu_err, p_low = np.loadtxt(sn_file, usecols=(1,2,3,4), unpack=True)
    Wmat = np.loadtxt(wmat_file)
    return names, z, mu, mu_err, p_low, Wmat

def unionChi2(omega_m, omega_lambda, z, mu, Wmat, omega_r=0):
    """
    Calculates the chi squared between the union2.1 data and a given 
    cosmological model.

    Parameters
    ----------
    omega_m : float
        Cosmological density for matter (baryonic+DM).
    omega_lambda: float
        Cosmological density for Lambda (assumed as a cosmological constant term).
    z : array-like
        Redshift data from union2.1.
    mu : array-like
        Distance modulus from union2.1.
    Wmat : array-like
        Covariance matrix from union2.1.
    omega_r : float, optional
        Cosmological density for radiation.

    Returns
    -------
    chi2 : ndarray
        Chi squared value of given parameters.

    Raises
    ------
    NonPhysicalCosmologyError
        If given densities do not correspond to a physically valid universe solution.

    Notes
    -----
    Considers that the ``Wmat`` covariance matrix is already corrected to eliminate nuisance
    parameters that could interfere with a chi2 estimation. In this case, mainly the effect
    of a low-mass galaxy host for the SNe and the normalization constant for the distance
    modulus mainly due to the effect of a chosen Hubble constant.
    
    """
    _, _, _, _, _, _, _, _, Mu, _ = cosmo.FLRW(z, omega_m, omega_lambda, omega_r=omega_r)
    residual = Mu - mu
    chi2 = residual @ Wmat @ residual
    return chi2

def unionPost(lnPrior, omega_m, omega_lambda, z, mu, Wmat, omega_r=0, lnEvid=0):
    """
    Calculates the bayesian log posterior of the cosmological model given the 
    observational data provided.

    Parameters
    ----------
    lnPrior : float
        Log prior of the selected cosmological parameters.
    omega_m : float
        Cosmological density for matter (baryonic+DM).
    omega_lambda: float
        Cosmological density for Lambda (assumed as a cosmological constant term).
    z : array-like
        Redshift data from union2.1.
    mu : array-like
        Distance modulus from union2.1.
    Wmat : array-like
        Covariance matrix from union2.1.
    omega_r : float, optional
        Cosmological density for radiation.
    lnEvid : float, optional
        Log evidence as a normalization of the bayesian probability. Is not necessary
        for an optimization algorithm or sampling method such as MCMC.

    Returns
    -------
    lnPost : float
        Calculated log posterior value of the given parameters conditioned on the data.

    Raises
    ------
    NonPhysicalCosmologyError
        If given densities do not correspond to a physically valid universe solution.

    Notes
    -----
    If ``lnPrior`` is chosen as ``-np.inf`` returns a ``-np.inf`` posterior, that can 
    be used to exclude a parameter set from a bayesian analysis, since it is equivalent
    to a posterior of zero.
    """
    lnLike = -0.5 * unionChi2(omega_m, omega_lambda, z, mu, Wmat, omega_r=omega_r, use_sys=use_sys)
    lnPost = lnLike + lnPrior - lnEvid
    return lnPost

def makeUnionLogPos(z, mu, Wmat):
    """
    Closure wrapper to create the log probability function to be sampled.

    Parameters
    ----------
    z : array-like
        Redshift data from union2.1.
    mu : array-like
        Distance modulus from union2.1.
    Wmat : array-like
        Covariance matrix from union2.1.

    Returns
    -------
    lnProb : callable
        Function that receives a single position argument 
        ``theta`` = [``omega_m``, ``omega_lambda``]
        such that it can be used in the MCMC implementation.

    Notes
    -----
    The function lnProb is such that it defines the probability as ``-np.inf`` for 
    any choice of cosmological parameters that raise a NonPhysicalCosmologyError.
    """
    def lnProb(theta):
        # Defining the prior as a box of values in omega, without a flatness constraint
        if theta[0] >= 0 and theta[0] <= 2 and theta[1] >= -1 and theta[1] <= 3:
            lnPrior = 0.0 # Uniform prior inside the selected range
        else:
            lnPrior = -np.inf # Value outside of prior is rejected

        try:
            post = unionPost(lnPrior, theta[0], theta[1], z, mu, Wmat)
        except cosmo.NonPhysicalCosmologyError:
            return -np.inf # Non physical cosmologies are rejected

        return post
        
    return lnProb
    
