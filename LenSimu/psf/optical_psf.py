"""
Optical PSF

Here we create an optical PhaseScreen to describe the variation of the PSF
model due to the optics.

"""

import numpy as np

import galsim

from ..utils import parser


class optical:
    def __init__(self, config, lam):
        self._lam = lam

        # Read config
        if isinstance(config, dict):
            self._opt_config = config
        elif isinstance(config, str):
            self._opt_config = self._load_config(config)
        else:
            raise TypeError("config must be a dict or a path to config file.")

        self._load_external_info()

    def _load_config(self, config):
        """Load config

        Load the config file for the PSF and keep only the optical part.
        Check if all the needed values are presents.

        Parameters
        ----------
        config_path: str
            Path to the Yaml config file.

        Return
        ------
        config: dict
            Dictionnary parsed from the config file.

        """

        if isinstance(config, str):
            config_dict = parser.ReadConfig(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise ValueError(
                "config must be a path to a config Yaml file or an instanciate"
                " dictionary."
            )

        parser._check_config(config_dict, parser._config_atmo_template)

        return config_dict["telescope"]

    def _load_external_info(self):
        self._e1_opt_arr = np.load(self._opt_config["focal_plane_file"]["e1"])
        self._e2_opt_arr = np.load(self._opt_config["focal_plane_file"]["e2"])
        size_opt = np.load(self._opt_config["focal_plane_file"]["size"])
        # NEED TO UPDATE LATER
        size_opt = np.sqrt(size_opt / 2.0) * 2.355  # *0.187
        self._size_factor_opt_arr = size_opt / np.mean(size_opt)

        self._config_focal_plane = {}
        self._config_focal_plane["size_img_x"] = (
            self._opt_config["data_sec"][1] - self._opt_config["data_sec"][0]
        ) + 1

        self._config_focal_plane["size_img_y"] = (
            self._opt_config["data_sec"][3] - self._opt_config["data_sec"][2]
        ) + 1

        (
            self._config_focal_plane["n_pix_y"],
            self._config_focal_plane["n_pix_x"],
        ) = self._e1_opt_arr[0].shape

        self._config_focal_plane["pix_size_x"] = (
            self._config_focal_plane["size_img_x"]
            / self._config_focal_plane["n_pix_x"]
        )

        self._config_focal_plane["pix_size_y"] = (
            self._config_focal_plane["size_img_y"]
            / self._config_focal_plane["n_pix_y"]
        )

    def get_focal_plane_value(self, x, y, ccd_num, do_rot):
        pos_x_msp = int(
            (x - self._opt_config["data_sec"][0])
            / self._config_focal_plane["pix_size_x"]
        )
        if pos_x_msp < 0:
            pos_x_msp = 0
        if pos_x_msp >= self._config_focal_plane["n_pix_x"]:
            pos_x_msp = self._config_focal_plane["n_pix_x"] - 1

        pos_y_msp = int(
            (y - self._opt_config["data_sec"][2])
            / self._config_focal_plane["pix_size_y"]
        )
        if pos_y_msp < 0:
            pos_y_msp = 0
        if pos_y_msp >= self._config_focal_plane["n_pix_y"]:
            pos_y_msp = self._config_focal_plane["n_pix_y"] - 1

        if do_rot:

            def rot_func(x):
                return np.fliplr(np.rot90(np.rot90(x)))
        else:

            def rot_func(x):
                return np.fliplr(x)

        g1 = rot_func(self._e1_opt_arr[ccd_num, :, :])[pos_y_msp, pos_x_msp]
        g2 = rot_func(self._e2_opt_arr[ccd_num, :, :])[pos_y_msp, pos_x_msp]
        size_factor = rot_func(self._size_factor_opt_arr[ccd_num, :, :])[
            pos_y_msp, pos_x_msp
        ]

        return g1, g2, size_factor

    def init_optical(
        self,
        pupil_bin=16,
        pad_factor=4,
        gsparams=None,
    ):
        if "pupil_plane_image" in self._opt_config.keys():
            self.aper, self.opt_psf = aperture_from_img(
                self._opt_config["pupil_plane_image"],
                self._lam,
                self._opt_config["aperture"]["diam"],
                self._opt_config["aperture"]["obscuration"],
                pupil_bin,
                gsparams,
            )
        else:
            self.aper, self.opt_psf = aperture_from_galsim(
                self._lam,
                self._opt_config["aperture"]["diam"],
                self._opt_config["aperture"]["obscuration"],
                self._opt_config["aperture"]["nstruts"],
                self._opt_config["aperture"]["strut_thick"],
                pad_factor,
                gsparams,
            )

    def get_optical_psf(self, pupil_bin=16, pad_factor=4, gsparams=None):
        self.init_optical(
            pupil_bin=pupil_bin,
            pad_factor=pad_factor,
            gsparams=gsparams,
        )
        tot_opt_psf = self.opt_psf

        return tot_opt_psf


def _aperture_from_img(pupil_path, lam, diam, obscur, pupil_bin, gsparams):
    pupil_plane_im = galsim.fits.read(pupil_path)

    pupil_plane_im = pupil_plane_im.bin(pupil_bin, pupil_bin)
    aper = galsim.Aperture(
        lam=lam,
        diam=diam,
        obscuration=obscur,
        pupil_plane_im=pupil_plane_im,
        gsparams=gsparams,
    )

    opt_psf = galsim.OpticalPSF(
        diam=diam,
        lam=lam,
        aper=aper,
        gsparams=gsparams,
    )

    return aper, opt_psf


def _aperture_from_galsim(
    lam, diam, obscur, nstruts, strut_thick, pad_factor, gsparams
):

    aper = galsim.Aperture(
        lam=lam,
        diam=diam,
        obscuration=obscur,
        nstruts=nstruts,
        strut_thick=strut_thick,
        pad_factor=pad_factor,
        gsparams=gsparams,
    )

    opt_psf = galsim.OpticalPSF(
        diam=diam,
        lam=lam,
        aper=aper,
        gsparams=gsparams,
    )

    return aper, opt_psf


# The last aperture is cached to save time in the execution
aperture_from_img = galsim.utilities.LRU_Cache(_aperture_from_img)
aperture_from_galsim = galsim.utilities.LRU_Cache(_aperture_from_galsim)
