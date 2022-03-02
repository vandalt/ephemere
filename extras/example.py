# %%
from importlib import reload
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table

# Import like this to be ablet to use reload
from ephemere import archive as earch
from ephemere import rmplanets as rmp

# %%
lbl_demo_dir = Path("/home/vandal/Documents/spirou/data/lbl_demo/")

OBJ_KEY = "OBJECT"

# %%
files = list(lbl_demo_dir.glob("lbl_Gl*_drift.rdb"))
files += list(lbl_demo_dir.glob("lbl_GL*_drift.rdb"))
files += list(lbl_demo_dir.glob("lbl_GJ*_drift.rdb"))
skip_objs = [
    "GJ 15 A",  # One planet is controversial, other is long period and poorly constrained
    "GJ 876",  # Multi-planets, large amplitude
    "GJ 3470",  # Period has big uncertainty so signal is poolry constrained
    "GJ 849",  # Period has big uncertainty so signal is poolry constrained
    "GJ 317",  # Big uncertainty on w
    "GJ 338",  # Big uncertainty on orbit
    "GJ 480",  # Big uncertainty on orbit
]


tbls = [Table.read(f) for f in files]
objs = [tbl[0][OBJ_KEY] for tbl in tbls]


# # Include lbl2 when correction is applied
# files_to_correct = list(lbl_demo_dir.glob("lbl*_drift.rdb"))

# %%
# TODO: Have way to resolve aliases either though masterfile or in a function/API here
# Masterfile issue: https://github.com/AntoineDarveau/exofile/issues/29
reload(earch)
archive_names = earch.get_archive_names(objs)
archive_names = [aname for aname in archive_names if aname not in skip_objs]
xfile = earch.load_archive(query=True)
xfile = xfile[xfile["hostname"].isin(archive_names)]
xfile.set_index(["hostname", "pl_letter"], inplace=True)
xfile.sort_index(inplace=True)

# %%
data_dict = dict(zip(archive_names, tbls))

# %%
# Loop over planets
reload(rmp)
orbit_params_scal = dict()
orbit_params_arr = dict()
rv_scal = dict()
rv_arr = dict()
rv_samples = dict()
times = dict()
for (hostname, pl_letter), row in xfile.iterrows():

    # hostname + pl_letter instead of pl_name in case name does not use same host alias
    pl_key = " ".join([hostname, pl_letter])
    orbit_params_scal[pl_key] = rmp.get_orbit_params(row, n_samples=0)

    # hostname + pl_letter instead of pl_name in case name does not use same host alias
    orbit_params_arr[pl_key] = rmp.get_orbit_params(row, n_samples=10_000)

    tbl = data_dict[hostname]

    t = np.linspace(
        orbit_params_scal[pl_key]["pl_orbtper"],
        orbit_params_scal[pl_key]["pl_orbtper"]
        + 5 * orbit_params_scal[pl_key]["pl_orbper"],
        num=1000,
    )
    times[pl_key] = t
    # t = np.linspace(
    #     tbl["BJD"].min(),
    #     tbl["BJD"].max(),
    #     num=1000,
    # )
    # times[pl_key] = t

    print(f"Gettting {pl_key} RVs")
    rv_scal[pl_key] = rmp.get_rv_signal(t, orbit_params_scal[pl_key])
    # orbit_params_arr[pl_key]["pl_orbtper"] = np.full_like(
    #     orbit_params_arr[pl_key]["pl_orbtper"], orbit_params_scal[pl_key]["pl_orbtper"]
    # )
    rv_samples[pl_key] = rmp.get_rv_signal(
        t, orbit_params_arr[pl_key], return_samples=True
    )
    rv_arr[pl_key] = rmp.rv_model_from_samples(rv_samples[pl_key])

    # plt.errorbar(
    #     tbl["BJD"],
    #     tbl["vrad"] - np.median(tbl["vrad"]),
    #     yerr=tbl["svrad"],
    #     fmt="ko",
    #     capsize=2,
    # )
    plt.plot(times[pl_key], rv_scal[pl_key], "bo")
    plt.plot(times[pl_key], rv_samples[pl_key][::10].T, alpha=0.01, color="r")
    plt.plot(times[pl_key], rv_arr[pl_key][:, 0], color="C0")
    # plt.fill_between(
    #     times[pl_key],
    #     rv_arr[pl_key][:, 0] + 3 * rv_arr[pl_key][:, 1],
    #     rv_arr[pl_key][:, 0] - 3 * rv_arr[pl_key][:, 1],
    #     alpha=0.5, color="C0"
    # )
    plt.show()
