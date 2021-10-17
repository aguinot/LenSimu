"""
Parser for the config files
"""

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

    stream = open(config_path, 'r')
    config = load(stream, Loader=Loader)

    return config
