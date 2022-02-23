import numpy as np
from pandas import Series
from radvel.kepler import rv_drive
from radvel.orbit import timetrans_to_timeperi
from scipy.stats import norm, truncnorm

# Keys from the archive
PER_KEY = "pl_orbper"
TP_KEY = "pl_orbtper"
TC_KEY = "pl_tranmid"
ECC_KEY = "pl_orbeccen"
OMEGA_KEY = "pl_orblper"
K_KEY = "pl_rvamp"
TRANSIT_FLAG = "tran_flag"

# Keys used to model orbit
ORB_KEYS = [PER_KEY, TP_KEY, ECC_KEY, OMEGA_KEY, K_KEY]
ORB_KEYS_REFS = [ok + "_reflink" for ok in ORB_KEYS]


def get_tp_param(planet: Series, n_samples: int = 0):
    # Get tp from tp or tc (transit)

    # If both are NaN, we can't obtain tp
    has_tp = not np.isnan(planet[TP_KEY])
    has_tc = not np.isnan(planet[TC_KEY])

    if not (has_tp or has_tc):
        raise ValueError(
            f"Both {TP_KEY} and {TC_KEY} are NaN f for {planet['pl_name']}"
        )

    if (planet[TRANSIT_FLAG] and has_tc) or not has_tp:
        # For transiting planets, Tc is usually better constained -> try first
        if n_samples > 0:
            tc = draw_param(planet, TC_KEY, n_samples)
        else:
            tc = planet[TC_KEY]
        # This assumes that planet[OTHER_PARAMS] are already distributions if they
        # need to be
        tp = timetrans_to_timeperi(
            tc, planet[PER_KEY], planet[ECC_KEY], planet[OMEGA_KEY]
        )
    else:
        if n_samples > 0:
            tp = draw_param(planet, TP_KEY, n_samples)
        else:
            tp = planet[TP_KEY]

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

    pval = planet[key]
    err = get_param_error(planet, key)

    if np.any(np.isnan([pval, err])):
        raise ValueError(f"Value and error for {key} must not be NaN")

    if key in [ECC_KEY, PER_KEY, K_KEY]:
        # Truncated normal if unphysical below 0
        upper = (1.0 - pval) / err if key == ECC_KEY else np.inf
        a, b = (0 - pval) / err, upper
        dist = truncnorm(a, b, loc=pval, scale=err)
    else:
        dist = norm(loc=pval, scale=err)

    return dist.rvs(ndraws)


def get_param_error(planet: Series, pkey: str):
    return np.mean(np.abs([planet[pkey + f"err{i}"] for i in (1, 2)]))



def get_orbit_params(planet: Series, n_samples: int = 0):

    orbpars = planet.copy()

    # Check that parameters are not missing
    special_cases = [TP_KEY]  # Parameters that we do not check directly
    regular_params = [p for p in ORB_KEYS if p not in special_cases]
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
    orbpars[TP_KEY] = get_tp_param(orbpars)

    # Keep only keys that we use to calculate orbit
    orbpars = orbpars[ORB_KEYS]

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
