import warnings

import pytest
from astropy.utils.diff import report_diff_values
from exofile.archive import ExoFile
from exofile.table_custom import Table
from pandas.core.frame import DataFrame

from ephemere.archive import get_archive_names, load_archive
from ephemere.constants import CONTROV_FLAG, OMEGA_KEY


def test_archive_names():
    input_list = [
        "GJ 1234",
        "GJ1234",
        "GL 111",
        "GL111A",
        "GL 111 B",
    ]

    expected_archive = [
        "GJ 1234",
        "GJ 1234",
        "GJ 111",
        "GJ 111 A",
        "GJ 111 B",
    ]

    archive_list = get_archive_names(input_list)

    assert archive_list == expected_archive


@pytest.fixture(scope="module")
def archive_df() -> DataFrame:
    return load_archive()


@pytest.fixture(scope="module")
def archive_tbl() -> Table:
    return load_archive(return_pandas=False)


@pytest.fixture(scope="module")
def xfile() -> Table:
    xfile = ExoFile.load(
        query=True,
        param=None,
        use_alt_file=False,
        warn_units=False,
        warn_local_file=False,
    )
    return xfile


@pytest.fixture(scope="module")
def archive_no_edits():
    return load_archive(keep_controv=True, convert_omega=False, return_pandas=False)


def test_load_archive(archive_df, archive_tbl):

    assert isinstance(archive_df, DataFrame)

    assert isinstance(archive_tbl, Table)

    assert archive_tbl.to_pandas().equals(archive_df)


def test_omega_units(archive_tbl):

    assert archive_tbl[OMEGA_KEY].unit == "rad"


def test_archive_exofile(xfile, archive_no_edits):

    # This returns true if they are identical
    assert report_diff_values(xfile, archive_no_edits)


def test_load_archive_warnings():

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        load_archive()


def test_archive_controv(archive_df, xfile, archive_no_edits):

    assert (archive_df[CONTROV_FLAG] == 0).all()

    xfile = xfile.to_pandas()
    archive_no_edits = archive_no_edits.to_pandas()
    assert archive_no_edits[CONTROV_FLAG].equals(xfile[CONTROV_FLAG])

    if (xfile[CONTROV_FLAG] == 1).any():
        assert not archive_df[CONTROV_FLAG].equals(xfile[CONTROV_FLAG])
