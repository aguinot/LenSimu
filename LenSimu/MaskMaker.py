import re

import numpy as np

from regions import PixCoord, PolygonPixelRegion
import galsim

from .utils import parser


class MaskMaker(object):
    def __init__(
        self,
        config,
        image,
        header,
        star_catalog,
        weight=None,
        kind="exp",
    ):
        self.image = image
        self.weight = weight
        self.header = header
        self.star_cat = star_catalog
        self._mask_config = self._load_config(config)
        self.kind = kind

    def _load_config(self, config):
        """Load config

        Load the config file for the PSF and keep only the atmospheric part.
        Check if all the needed values are presents.

        Parameters
        ----------
        config_path: str
            Path to the Yaml config file.

        Return
        ------
        config: dict
            Dictionary parsed from the config file.

        """

        if isinstance(config, str):
            config_dict = parser.ReadConfig(config)
        elif isinstance(config, dict):
            config_dict = config
        else:
            raise ValueError(
                "config must be a path to a config Yaml file or an instantiate"
                " dictionary."
            )

        parser._check_config(config_dict, parser._config_atmo_template)

        return config_dict["masking"]

    def zero_weight(self):
        md_mask = np.zeros(self.weight.array.shape, dtype=np.uint32)
        md_mask[self.weight.array == 0] = 1

        md_mask *= self._mask_config["weight_mask"]["flag_value"]

        return md_mask

    def border(self):
        width = self._mask_config["border"]["border_size"]
        ccd_bounds_ = np.array(
            re.findall(r"\d+", self.header["DETSIZE"]), dtype=int
        )

        border_bounds = galsim.BoundsI(
            ccd_bounds_[0] + width,
            ccd_bounds_[1] - width,
            ccd_bounds_[2] + width,
            ccd_bounds_[3] - width,
        )
        stamp_bounds = self.image.bounds

        mask_img = galsim.ImageUS(stamp_bounds, init_value=1)
        cmon_bounds = border_bounds & stamp_bounds
        if cmon_bounds.isDefined():
            mask_img[cmon_bounds].fill(0)
        border_mask = (
            mask_img.array * self._mask_config["border"]["flag_value"]
        )
        return border_mask.astype(np.uint32)

    def star_mask(
        self,
        types="HALO",
    ):
        """Create Mask.

        Apply mask from model to stars and save into DS9 region file.

        NOTE: This function comes from ShapePipe mask_package.
        See: https://github.com/CosmoStat/shapepipe/blob/develop/shapepipe/modules/mask_package/mask.py  # noqa

        Parameters
        ----------
        stars : dict
            Stars dictionary (output of ``find_stars``)
        types : {'HALO', 'SPIKE'}, optional
            Type of mask, options are ``HALO`` or ``SPIKE``
        mag_limit : float, optional
            Faint magnitude limit for mask, default is ``18.0``
        mag_pivot : float, optional
            Pivot magnitude for the model, default is ``13.8``
        scale_factor : float, optional
            Scaling for the model, default is ``0.3``

        Raises
        ------
        ValueError
            If no star catalogue is provided
        ValueError
            If an invalid option is provided for type

        """

        if types == "HALO":
            config = self._mask_config["halo_mask"]
        elif types == "SPIKE":
            config = self._mask_config["spike_mask"]
        else:
            raise ValueError('Mask types need to be in ["HALO", "SPIKE"]')

        # Load parameters
        mag_limit = config["mag_limit"]
        mag_pivot = config["mag_pivot"]
        scale_factor = config["scale_factor"]
        mask_model = np.loadtxt(config["reg_path"]).T

        # Precompute model
        m_mag = self.star_cat["J_mag"] < mag_limit
        scaling = 1 - scale_factor * (
            self.star_cat["J_mag"][m_mag] - mag_pivot
        )
        pos = self.image.wcs.astropy.all_world2pix(
            self.star_cat["ra"][m_mag],
            self.star_cat["dec"][m_mag],
            0,
        )

        # Init flag image
        img_shape = self.image.array.shape
        star_mask = np.zeros(img_shape, dtype=np.uint32)

        for i in range(sum(m_mag)):
            angle = np.arctan2(mask_model[1], mask_model[0])
            ll = scaling[i] * np.sqrt(mask_model[0] ** 2 + mask_model[1] ** 2)
            xnew = ll * np.cos(angle)
            ynew = ll * np.sin(angle)

            poly = PolygonPixelRegion(
                PixCoord(pos[0][i] + xnew + 0.5, pos[1][i] + ynew + 0.5)
            )
            mask_tmp = poly.to_mask().to_image(img_shape, dtype=np.uint32)
            if mask_tmp is None:
                continue
            mask_tmp *= config["flag_value"]
            star_mask[star_mask == 0] += mask_tmp[star_mask == 0]

        return star_mask

    def get_mask(self):
        # mask = np.zeros(self.image.array.shape, dtype=int)
        mask = galsim.ImageUI(self.image.bounds, init_value=0)
        # weight mask
        if self._mask_config["weight_mask"]["make"] & (self.kind == "exp"):
            if self.weight is None:
                raise ValueError("weight is not defined.")
            mask += self.zero_weight()

        # border mask
        if self._mask_config["border"]["make"] & (self.kind == "exp"):
            mask += self.border()

        # halo mask
        if self._mask_config["halo_mask"]["make"]:
            mask += self.star_mask(types="HALO")

        # spike mask
        if self._mask_config["spike_mask"]["make"]:
            mask += self.star_mask(types="SPIKE")

        return mask
