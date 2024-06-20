"""
Parser for the config files
"""

import numpy as np

from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


def ReadConfig(config_path):
    """Read config file

    Parse Yaml file to a dictionnary

    Parameters
    ----------
    config_path: str
        Path to the config file.

    Return
    ------
    config: dict
        Dictionnary parsed from Yaml file.

    """

    with open(config_path, "r") as stream:
        config = load(stream, Loader=Loader)

    return config


def _check_config(config, config_template):
    """Check config

    Check the consistancy of the provided configuration based on a config
    template.

    Parameters
    ----------
    config: dict
        Configuration dictionary.
    config_template: dict
        Structure template to match

    """

    check = np.product(
        [key in list(config.keys()) for key in list(config_template.keys())]
    )

    if not check:
        raise ValueError(
            f"{list(config_template.keys())} sections not found in config."
            + f" Got {list(config.keys())}"
        )
    for key in config_template.keys():
        if isinstance(config_template[key], dict):
            _check_config(config[key], config_template[key])


# NOTE: There is probably a better way to do that
_config_atmo_template = {
    "atmospheric": {
        "HV_coeff": {
            "A0": None,
            "H0": None,
            "A1": None,
            "H1": None,
            "A2": None,
            "H2": None,
            "A3": None,
            "H3": None,
            "d": None,
        },
        "L0_values": {
            "alts": None,
            "L0": None,
            "spread": None,
        },
        "wind": {
            "GW_coeff": {
                "H": None,
                "T": None,
            },
            "wind_speed": {
                "ground": None,
                "trop": None,
            },
        },
    },
    "telescope": {
        "FOV": None,
        "aperture": None,
    },
}
