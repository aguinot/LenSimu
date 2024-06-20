from tqdm import tqdm
import os

import numpy as np
import pandas as pd

import galsim

from astropy.wcs import WCS
from astropy import coordinates as coord
from astropy import units as u
from astropy.io import fits

from .utils import parser
from .utils.header_builder import make_header
from .utils.catalog import write_catalog
from .CCDMaker import CCDMaker
from .psf.atmospheric import seeing_distribution


N_CCD = 40


class ExposureMaker(object):
    """ """

    def __init__(self, expnum, config, gal_catalog, star_catalog):
        self.expnum = expnum
        self.config_path = config
        (self._atm_config, self._opt_config, self._file_config) = (
            self._load_config(config)
        )
        self._init_catalog(gal_catalog, star_catalog)

        self._init_output()

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

        return (
            config_dict["atmospheric"],
            config_dict["telescope"],
            config_dict["file"],
        )

    def _init_catalog(self, gal_catalog, star_catalog):
        """ """

        self.header_info = pd.read_pickle(self._opt_config["header_info"]).loc[
            self.expnum
        ]

        gain = np.load(self._opt_config["gain"])

        self.header_list = make_header(
            self.header_info,
            gain,
            self._opt_config["ccd_size"],
            self._opt_config["data_sec"],
        )

        gal_catalog_ap = coord.SkyCoord(
            ra=gal_catalog["ra"] * u.degree, dec=gal_catalog["dec"] * u.degree
        )
        star_catalog_ap = coord.SkyCoord(
            ra=star_catalog["ra"] * u.degree,
            dec=star_catalog["dec"] * u.degree,
        )

        field_center = coord.SkyCoord(
            ra=self.header_info.loc[0]["CRVAL1"] * u.degree,
            dec=self.header_info.loc[0]["CRVAL2"] * u.degree,
        )

        # Pre-select objects
        # m_gal = gal_catalog_ap.search_around_sky(
        #     field_center,
        #     seplimit=self._opt_config["FOV"]*2.*u.degree
        # )[1]
        # m_star = star_catalog_ap.search_around_sky(
        #     field_center,
        #     seplimit=self._opt_config["FOV"]*2.*u.degree
        # )[1]
        m_gal = (
            gal_catalog_ap.separation(field_center)
            < self._opt_config["FOV"] * 1.0 * u.degree
        )
        m_star = (
            star_catalog_ap.separation(field_center)
            < self._opt_config["FOV"] * 1.0 * u.degree
        )

        self.gal_catalog = gal_catalog[m_gal]
        self.star_catalog = star_catalog[m_star]
        self.gal_catalog_ap = gal_catalog_ap[m_gal]
        self.star_catalog_ap = star_catalog_ap[m_star]

    def _init_output(self):
        """ """

        self.output_image_path = self._file_config["output_dir"] + "/images"
        self.output_catalog_path = (
            self._file_config["output_dir"] + "/catalogs"
        )

        if not os.path.exists(self.output_image_path):
            os.mkdir(self.output_image_path)
        if not os.path.exists(self.output_catalog_path):
            os.mkdir(self.output_catalog_path)

    def get_ccd_catalog(self, ccd_number):
        """ """

        header_ori = self.header_list[ccd_number].copy()

        self.ccd_wcs = WCS(header_ori)

        wcs_foot = WCS(naxis=2)
        wcs_foot.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs_foot.wcs.cd = np.array(
            [
                [header_ori["CD1_1"], header_ori["CD1_2"]],
                [header_ori["CD2_1"], header_ori["CD2_2"]],
            ]
        )
        wcs_foot.wcs.crval = np.array(
            self.ccd_wcs.all_pix2world(
                header_ori["NAXIS1"] / 2,
                header_ori["NAXIS2"] / 2,
                1,
            )
        )
        naxis1 = header_ori["NAXIS1"] + 1_500
        naxis2 = header_ori["NAXIS2"] + 1_500
        wcs_foot.wcs.crpix = np.array([naxis1 / 2, naxis2 / 2])
        wcs_foot.array_shape = np.array([naxis2, naxis1])

        # mask_gal = self.ccd_wcs.footprint_contains(self.gal_catalog_ap)
        # mask_star = self.ccd_wcs.footprint_contains(self.star_catalog_ap)
        mask_gal = wcs_foot.footprint_contains(self.gal_catalog_ap)
        mask_star = wcs_foot.footprint_contains(self.star_catalog_ap)

        return self.gal_catalog[mask_gal], self.star_catalog[mask_star]

    def running_func(self, ccd_number, g1, g2, target_seeing, seed):
        """ """

        gal_catalog, star_catalog = self.get_ccd_catalog(ccd_number)

        header = self.header_list[ccd_number]
        header["SEEING"] = np.round(target_seeing, 3)

        ccd_obj = CCDMaker(
            self.config_path,
            ccd_number,
            galsim.AstropyWCS(wcs=self.ccd_wcs),
            self.header_info["background"][ccd_number],
            self.header_info["EXPTIME"][ccd_number],
            self.header_info["PHOTZP"][ccd_number],
            g1,
            g2,
            target_seeing,
            seed,
        )

        ccd_img, ccd_catalog = ccd_obj.go(
            gal_catalog,
            star_catalog,
        )

        return [ccd_img, header], ccd_catalog

    def runner(self, g1, g2, target_seeing, seed_ori=1234):
        """ """

        single_exposure = []
        single_exposure_cat = []
        # single_exposure_psf_cat = []

        for ccd_number in tqdm(range(0, N_CCD), total=N_CCD):
            seed = seed_ori  # + 10000*ccd_number

            ccd_tmp, cat_tmp = self.running_func(
                ccd_number,
                g1,
                g2,
                target_seeing,
                seed,
            )
            single_exposure.append(ccd_tmp)
            single_exposure_cat.append(cat_tmp)
            # single_exposure_psf_cat.append(psf_cat_tmp)

        return single_exposure, single_exposure_cat  # ,single_exposure_psf_cat

    def write_output(self, ccd_images, ccd_cats):  # , psf_cats):
        """ """

        ori_name = self.expnum

        # Write images
        primary_hdu = fits.PrimaryHDU()
        hdu_list = fits.HDUList([primary_hdu])
        for i in range(N_CCD):
            saturate = self._opt_config["saturate"]
            img_tmp = np.copy(ccd_images[i][0].array)
            # Check saturation
            img_tmp[img_tmp > saturate] = saturate

            # Convert to 16bit
            img_tmp = img_tmp.astype(np.int16)
            img_tmp[img_tmp < 0] = np.max(img_tmp)

            hdu_list.append(
                fits.CompImageHDU(
                    img_tmp, header=ccd_images[i][1], name="CCD_{}".format(i)
                )
            )
        hdu_list.writeto(
            self.output_image_path + "/simu_image-{}.fits.fz".format(ori_name),
            overwrite=True,
        )

        # Write catalogs
        out_cat_name = self.output_catalog_path + "/simu_cat-{}.fits".format(
            ori_name
        )
        write_catalog(ccd_cats, out_cat_name)

        # # Write PSF catalogs
        # out_psf_cat = io.FITSCatalog(
        #     self.output_catalog_path + '/simu_psf_cat-{}.fits'.format(
        #         ori_name
        #     ),
        #     pen_mode=io.BaseCatalog.OpenMode.ReadWrite)
        # for i in range(self.param_dict['TELESCOPE']['N_CCD']):
        #     out_psf_cat.save_as_fits(
        #         psf_cats[i], ext_name='CCD_{}'.format(i)
        #     )

    def go(self, g1, g2, target_seeing=None, seed=1234):
        """ """

        if target_seeing is None:
            seeing_dist = seeing_distribution(
                self._atm_config["seeing_distribution"],
                seed,
            )
            target_seeing = seeing_dist.get(1)[0]
        elif not isinstance(target_seeing, float):
            raise ValueError("target_seeing must be float.")

        ccd_img, ccd_cat = self.runner(g1, g2, target_seeing, seed)
        self.write_output(ccd_img, ccd_cat)  # , psf_cat)
