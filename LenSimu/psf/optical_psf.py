"""
Optical PSF

Here we create an optical PhaseScreen to describe the variation of the PSF
model due to the optics.

"""

import numpy as np

import galsim

from ..utils import parser
from .optical_psf_nb import rescale_focal_plane


class optical:
    def __init__(self, config, lam, fixed_gauss_size=0.42):
        self._lam = lam
        self._fixed_gauss_size = fixed_gauss_size

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

    def init_optical(self, atm_psf=None, sig_atm=None, **kwargs):
        self.aper = galsim.Aperture(
            lam=self._lam,
            screen_list=atm_psf,
            **self._opt_config["aperture"],
            **kwargs,
        )

        self.opt_psf = galsim.OpticalPSF(
            diam=self._opt_config["aperture"]["diam"],
            lam=self._lam,
            aper=self.aper,
        )

        if sig_atm is not None:
            sig_opt_diff = self.opt_psf.calculateMomentRadius()
            self._sig_opt = np.sqrt(
                (self._fixed_gauss_size / 2.355) ** 2 + sig_opt_diff**2
            )
            # self._rescale_focal_plane(sig_atm)
            self._e1_opt_arr, self._e2_opt_arr = rescale_focal_plane(
                sig_atm,
                self._sig_opt,
                self._e1_opt_arr,
                self._e2_opt_arr,
            )

    def get_optical_psf(self, x, y, ccd_num, do_rot):
        fp_g1, fp_g2, fp_size_factor = self.get_focal_plane_value(
            x,
            y,
            ccd_num,
            do_rot,
        )

        fp_psf = galsim.Gaussian(
            fwhm=self._fixed_gauss_size * fp_size_factor
        ).shear(g1=fp_g1, g2=fp_g2)

        tot_opt_psf = galsim.Convolve((fp_psf, self.opt_psf))

        return tot_opt_psf
