from pathlib import Path
from typing import List, Optional, Union

from exofile.archive import ExoFile
from pandas.core.frame import DataFrame

from ephemere import constants as const


def load_archive(
    query: bool = True,
    exofile_param_file: Optional[Union[str, Path]] = None,
    use_alt_exofile: bool = False,
    keep_controv: bool = False,
    warn_units: bool = False,
    warn_local_file: bool = False,
    convert_omega: bool = True,
    return_pandas: bool = True,
    **kwargs
) -> DataFrame:

    # TODO: Either merge the warn_units in exofile or use warning filters here instead
    # Masterfile PR: https://github.com/AntoineDarveau/exofile/pull/26
    tbl = ExoFile.load(
        query=query,
        param=exofile_param_file,
        use_alt_file=use_alt_exofile,
        warn_units=warn_units,
        warn_local_file=warn_local_file,
        **kwargs,
    )

    if not keep_controv:
        tbl = tbl[tbl[const.CONTROV_FLAG] == 0]

    # All our RV calculations expect omega in radians, so convert now
    if convert_omega and tbl[const.OMEGA_KEY].unit != "rad":
        tbl[const.OMEGA_KEY] = tbl[const.OMEGA_KEY].to("rad")
        tbl[const.OMEGA_KEY + "err1"] = tbl[const.OMEGA_KEY + "err1"].to("rad")
        tbl[const.OMEGA_KEY + "err2"] = tbl[const.OMEGA_KEY + "err2"].to("rad")

    return tbl.to_pandas() if return_pandas else tbl


def get_archive_names(names: List[str]) -> List[str]:
    new_objs = names.copy()

    # GL/GJ stars are all "GJ " in NASA archive (and in exofile)
    def _replace_gj(oname: str):
        # Version with spaces before otherwise the space-free version will match
        gj_alts = ["GJ", "GL", "Gl"]
        gj_alts_with_space = [gja + " " for gja in gj_alts]
        gj_alts = tuple(gj_alts_with_space + gj_alts)

        if oname.startswith(gj_alts):
            for gja in gj_alts:
                if oname.startswith(gja):
                    return oname.replace(gja, "GJ ")
        else:
            return oname

    new_objs = [_replace_gj(o) for o in new_objs]

    # Handle binary stars
    def _format_binary(oname: str):
        if oname.endswith((" A", " B")):
            return oname
        elif oname.endswith(("A", "B")):
            return oname[:-1] + " " + oname[-1]
        else:
            return oname

    new_objs = [_format_binary(o) for o in new_objs]

    return new_objs
