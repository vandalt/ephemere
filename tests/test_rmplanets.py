import numpy as np
import pandas as pd
import pytest
from radvel.orbit import timetrans_to_timeperi

import ephemere.constants as const
import ephemere.rmplanets as rmp


@pytest.fixture()
def myplanet() -> pd.Series:
    myplanet = pd.Series(
        {
            const.NAME_KEY: "coolstar b",
            const.TP_KEY: 1.0,
            const.TC_KEY: 2.0,
            const.PER_KEY: 5.0,
            const.ECC_KEY: 0.0,
            const.OMEGA_KEY: 0.0,
            const.K_KEY: 4.0,
            const.TRANSIT_FLAG: 0,
        }
    )
    # Add errors for sample test
    for k in const.ORB_KEYS + [const.TC_KEY]:
        myplanet[k + "err1"] = myplanet[k] * 0.1 if myplanet[k] > 0.0 else 0.01
        myplanet[k + "err2"] = myplanet[k] * 0.1 if myplanet[k] > 0.0 else 0.01

    return myplanet


def test_tp_param(myplanet):

    # If both are there but non-transiting, should use TP
    myplanet[const.TRANSIT_FLAG] = 0
    tp = rmp.get_tp_param(myplanet)
    assert tp == myplanet[const.TP_KEY]

    # If both are there and transiting, should use TC
    myplanet[const.TRANSIT_FLAG] = 1
    tp = rmp.get_tp_param(myplanet)
    tp_from_tc = timetrans_to_timeperi(
        myplanet[const.TC_KEY],
        myplanet[const.PER_KEY],
        myplanet[const.ECC_KEY],
        myplanet[const.OMEGA_KEY],
    )
    assert tp == tp_from_tc

    myplanet_no_tp = myplanet.copy()
    myplanet_no_tp[const.TP_KEY] = np.nan
    tp = rmp.get_tp_param(myplanet_no_tp)
    assert tp == tp_from_tc

    myplanet_no_tc = myplanet.copy()
    myplanet_no_tc[const.TC_KEY] = np.nan
    tp = rmp.get_tp_param(myplanet_no_tc)
    assert tp == myplanet[const.TP_KEY]

    myplanet_no_tcp = myplanet_no_tp.copy()
    myplanet_no_tcp[const.TC_KEY] = np.nan
    with pytest.raises(ValueError):
        rmp.get_tp_param(myplanet_no_tcp)

    tp = rmp.get_tp_param(myplanet, n_samples=20)
    assert isinstance(tp, np.ndarray)


def test_draw_shape(myplanet):

    samples = rmp.draw_param(myplanet, const.K_KEY, ndraws=1000)
    assert isinstance(samples, np.ndarray)
    assert samples.ndim == 1
    assert len(samples) == 1000


def test_draw_bounds(myplanet):

    # Use large errors and make sure cropped at 0 (and 1 for ecc)
    my_planet_large_err = myplanet.copy()
    my_planet_large_err[const.ECC_KEY + "err1"] = 1.0
    my_planet_large_err[const.ECC_KEY + "err2"] = 1.0
    my_planet_large_err[const.OMEGA_KEY + "err1"] = 10.0
    my_planet_large_err[const.OMEGA_KEY + "err2"] = 10.0
    my_planet_large_err[const.PER_KEY + "err1"] = (
        my_planet_large_err[const.PER_KEY] * 10.0
    )
    my_planet_large_err[const.PER_KEY + "err2"] = (
        my_planet_large_err[const.PER_KEY] * 10.0
    )
    my_planet_large_err[const.K_KEY + "err1"] = my_planet_large_err[const.K_KEY] * 10.0
    my_planet_large_err[const.K_KEY + "err2"] = my_planet_large_err[const.K_KEY] * 10.0

    ecc_samples = rmp.draw_param(my_planet_large_err, const.ECC_KEY, ndraws=1000)
    assert np.all((ecc_samples > 0.0) & (ecc_samples < 1.0))

    # Make sure can draw outside for other parameters
    w_samples = rmp.draw_param(my_planet_large_err, const.OMEGA_KEY, ndraws=1000)
    assert np.any(w_samples < 0.0)
    assert np.any(w_samples > 1.0)

    k_samples = rmp.draw_param(my_planet_large_err, const.K_KEY, ndraws=1000)
    assert np.all(k_samples > 0.0)

    per_samples = rmp.draw_param(my_planet_large_err, const.PER_KEY, ndraws=1000)
    assert np.all(per_samples > 0.0)


def test_draw_errors(myplanet):
    mykey = "mykey"
    myplanet[mykey] = np.nan
    myplanet[mykey + "err1"] = 1.0
    myplanet[mykey + "err2"] = 1.0

    with pytest.raises(
        ValueError, match=f"Value and error for {mykey} must not be NaN"
    ):
        rmp.draw_param(myplanet, mykey, ndraws=1000)

    myplanet[mykey] = 1.0
    myplanet[mykey + "err1"] = np.nan
    myplanet[mykey + "err2"] = np.nan

    with pytest.raises(
        ValueError, match=f"Value and error for {mykey} must not be NaN"
    ):
        rmp.draw_param(myplanet, mykey, ndraws=1000)

    with pytest.raises(KeyError):
        rmp.draw_param(myplanet, "fdfjkad", ndraws=1000)


def test_get_error(myplanet):

    mykey = "mykey"
    myplanet[mykey + "err1"] = 1.0
    myplanet[mykey + "err2"] = 2.0

    assert isinstance(rmp.get_param_error(myplanet, mykey), float)
    assert rmp.get_param_error(myplanet, mykey) == 1.5

    myplanet[mykey + "err2"] = -2.0
    assert rmp.get_param_error(myplanet, mykey) == 1.5


def test_get_orbit_params(myplanet):

    orbpars = rmp.get_orbit_params(myplanet)

    # Single value
    assert list(orbpars.index) == const.ORB_KEYS
    assert orbpars.equals(myplanet[const.ORB_KEYS])
    # Using apply because planet name was there initially so the pandas type is "object"
    assert (orbpars.apply(type) == float).all()

    # Multiple draws
    orbpars_samples = rmp.get_orbit_params(myplanet, n_samples=1000)
    assert (orbpars_samples.apply(type) == np.ndarray).all()
    assert (orbpars_samples.str.len() == 1000).all()


def test_get_rv_signal(myplanet):
    t = np.linspace(
        myplanet[const.TP_KEY],
        myplanet[const.TP_KEY] + 5 * myplanet[const.PER_KEY],
        num=1000,
    )

    # Single values return one RV curve
    params = rmp.get_orbit_params(myplanet)
    rv_curve = rmp.get_rv_signal(t, params)
    assert rv_curve.ndim == 1
    assert len(rv_curve) == len(t)
    assert len(rv_curve)

    # Multiple values return one draw per parameter set
    params = rmp.get_orbit_params(myplanet, n_samples=10_000)
    rv_samples = rmp.get_rv_signal(t, params, return_samples=True)
    assert rv_samples.ndim == 2
    assert rv_samples.shape == (10_000, len(t))

    # return_samples=False should return model and envelope like rv_model_from_samples
    assert np.all(rmp.get_rv_signal(t, params) == rmp.rv_model_from_samples(rv_samples))


def test_rv_from_samples(myplanet):
    t = np.linspace(
        myplanet[const.TP_KEY],
        myplanet[const.TP_KEY] + 5 * myplanet[const.PER_KEY],
        num=1000,
    )
    params = rmp.get_orbit_params(myplanet, n_samples=10_000)
    rv_samples = rmp.get_rv_signal(t, params, return_samples=True)

    rv_model = rmp.rv_model_from_samples(rv_samples)

    assert rv_model.ndim == 2
    assert rv_model.shape == (len(t), 2)
    assert np.all(rv_model[:, 0] == np.median(rv_samples, axis=0))
