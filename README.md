# ephemere

Get exoplanet RV signals from known planets using the NASA exoplanet Archive.

_ephemere_ uses [exofile](https://github.com/AntoineDarveau/exofile) to complement the
[NASA exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/) table. From
the parameters in this table, it generates RV signals as a function of time. It
can either use the main parameter value or generate random RV signals based on
the parameter uncertainties (assuming Gaussian uncertainties).

See the [example script](extras/example.py) for an example of how _ephemere_ can
be used to generate RV signals.

## Installation

To install the latest release, use 

```shell
python -m pip install ephemere
```

To install the development version, you can clone it from github and
install it with the following commands:
```shell
git clone https://github.com/vandalt/ephemere.git
cd ephemere
python -m pip install -U -e ".[dev]"
```
