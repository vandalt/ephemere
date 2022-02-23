from pathlib import Path
from typing import Optional, Union, List
from pandas.core.frame import DataFrame

from exofile.archive import ExoFile

CONTROV_FLAG = "pl_controv_flag"

def load_archive(
        query: bool = True,
        exofile_param_file: Optional[Union[str, Path]] = None,
        use_alt_exofile: bool = False,
        keep_controv: bool = False,
        warn_units=False,
        warn_local_file=False,
        **kwargs
    ) -> DataFrame:

    # TODO: Either merge the warn_units in exofile or use warning filters here instead
    tbl = ExoFile.load(
        query=query,
        param=exofile_param_file,
        use_alt_file=use_alt_exofile,
        warn_units=warn_units,
        warn_local_file=warn_local_file,
        **kwargs,
    )

    if not keep_controv:
        tbl = tbl[tbl[CONTROV_FLAG] == 0]

    return tbl.to_pandas()


def get_archive_names(names: List[str]):
    new_objs = names.copy()

    # GL/GJ stars are all "GJ " in NASA archive (and in exofile)
    def _replace_gj(oname):
        gj_alts = ("GJ ", "GJ", "GL ", "GL", "Gl ", "Gl")
        if oname.startswith(gj_alts):
            for gja in gj_alts:
                if oname.startswith(gja):
                    return oname.replace(gja, "GJ ")
        else:
            return oname

    new_objs = [_replace_gj(o) for o in new_objs]

    # Handle binary stars
    def _format_binary(oname):
        if oname.endswith((" A", " B")):
            return oname
        elif oname.endswith(("A", "B")):
            return oname[:-1] + " " + oname[-1]
        else:
            return oname

    new_objs = [_format_binary(o) for o in new_objs]

    return new_objs
