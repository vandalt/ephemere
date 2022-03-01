import numpy as np
from pandas import Series, DataFrame
# TODO: Silence celerite warnign on radvel import
from radvel.kepler import rv_drive
from radvel.orbit import timetrans_to_timeperi
from scipy.stats import norm, truncnorm

from ephemere import constants as const

def get_tp_param(planet: Series, n_samples: int = 0):
    # Get tp from tp or tc (transit)

    # If both are NaN, we can't obtain tp
    has_tp = not np.isnan(planet[const.TP_KEY])
    has_tc = not np.isnan(planet[const.TC_KEY])

    if not (has_tp or has_tc):
        raise ValueError(
            f"Both {const.TP_KEY} and {const.TC_KEY} are NaN f for {planet['pl_name']}"
        )

    if (planet[const.TRANSIT_FLAG] and has_tc) or not has_tp:
        # For transiting planets, Tc is usually better constained -> try first
        if n_samples > 0:
            tc = draw_param(planet, const.TC_KEY, n_samples)
        else:
            tc = planet[const.TC_KEY]
        # This assumes that planet[OTHER_PARAMS] are already distributions if they
        # need to be
        tp = timetrans_to_timeperi(
            tc, planet[const.PER_KEY], planet[const.ECC_KEY], planet[const.OMEGA_KEY]
        )
    else:
        if n_samples > 0:
            tp = draw_param(planet, const.TP_KEY, n_samples)
        else:
            tp = planet[const.TP_KEY]

    return tp


def draw_param(planet: Series, key: str, ndraws: int) -> np.ndarray:
    """
    Draw parameter values based on planet parameters and uncertainties.

    :param planet: Pandas series with planet parameters.
    :type planet: Series
    :param key: Key of the parameter to draw
    :type key: str
    :param ndraws: Number of draws.
    :type ndraws: int
    :return: Array of Monte-Carlo draws for the parameter.
    :rtype: np.ndarray
    """

    # NOTE: Maybe astropy uncertainties could do the job, but would need truncated normal
    # (related issue: https://github.com/astropy/astropy/issues/12886)
    # Bottom line: can implement custom distribution, but what we have does the job for now
    pval = planet[key]
    err = get_param_error(planet, key)

    if np.any(np.isnan([pval, err])):
        raise ValueError(f"Value and error for {key} must not be NaN")

    if key in [const.ECC_KEY, const.PER_KEY, const.K_KEY]:
        # Truncated normal if unphysical below 0
        upper = (1.0 - pval) / err if key == const.ECC_KEY else np.inf
        a, b = (0 - pval) / err, upper
        dist = truncnorm(a, b, loc=pval, scale=err)
    else:
        dist = norm(loc=pval, scale=err)

    return dist.rvs(ndraws)


def get_param_error(planet: Series, pkey: str):
    return np.mean(np.abs([planet[pkey + f"err{i}"] for i in (1, 2)]))



def get_orbit_params(planet: Series, n_samples: int = 0)  -> Series:

    orbpars = planet.copy()

    # Check that parameters are not missing
    special_cases = [const.TP_KEY]  # Parameters that we do not check directly
    regular_params = [p for p in const.ORB_KEYS if p not in special_cases]
    for pkey in regular_params:
        if np.isnan(orbpars[pkey]):
            raise ValueError(f"Parameter {pkey} for {orbpars['pl_name']} is NaN")

        # Draw parameters from normal (or truncated normal if P, ecc, K)
        # distribution to account for uncertainty
        err = get_param_error(orbpars, pkey)
        if n_samples > 0 and err != 0.0 and not np.isnan(err):
            orbpars[pkey] = draw_param(orbpars, pkey, n_samples)

    # TP might come from transit time, so handle separately
    # We use orpars, so uncertainties from draw_param are propagated
    orbpars[const.TP_KEY] = get_tp_param(orbpars, n_samples=n_samples)

    # Keep only keys that we use to calculate orbit
    orbpars = orbpars[const.ORB_KEYS]

    # If some parameters have scalars and others have array,
    # Repeat scalars to arrays of same length
    try:
        lvals = orbpars.str.len()
        plen = int(lvals.max())  # All non-zero should be max
        scalar_mask = lvals.isna()
        orbpars[scalar_mask] = orbpars[scalar_mask].apply(
            lambda x: np.full(plen, x)
        )
    except AttributeError:
        # if all scalars, nothing to do (just filter to dict)
        pass

    return orbpars


def rv_model_from_samples(rv_samples: np.ndarray) -> np.ndarray:

    # Get curve and 1-sigma enveloppe from rv draws
    rv_16th, rv_med, rv_84th = np.percentile(
        rv_samples, [16, 50, 84], axis=0
    )
    rv_err_lo = rv_med - rv_16th
    rv_err_hi = rv_84th - rv_med
    rv_err = np.mean([rv_err_hi, rv_err_lo], axis=0)

    return np.array([rv_med, rv_err]).T


def get_rv_signal(t: np.ndarray, params: Series, return_samples: bool = False) -> np.ndarray:

    t = np.atleast_1d(t)

    scalar_mask = params.apply(np.isscalar)

    if scalar_mask.any() and not scalar_mask.all():
        raise TypeError(
            "params must contain only scalars or only arrays, not a mix of both"
        )

    is_scalar = scalar_mask.all()

    # If not scalar, dataframe will be more convenient
    if not is_scalar:
        params = DataFrame(params.to_dict())

    # Make sure parameters are properly ordered for radvel
    # Keep after dict conversion because dicts are unordered
    params = params[const.ORB_KEYS]

    if is_scalar:
        orbel = params.to_list()
        rv = rv_drive(t, orbel)

        return rv

    # Store RVs for each parameter sample in n_samples x len(t) array
    rv_samples = np.empty((len(params), len(t)))
    for i, pseries in params.iterrows():
        orbel = pseries.to_list()
        rv_samples[i] = rv_drive(t, orbel)

    if return_samples:
        return rv_samples
    else:
        rv_model_from_samples(rv_samples)
