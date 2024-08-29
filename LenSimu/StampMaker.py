import os
import copy
# from tqdm import tqdm

import numpy as np
import pandas as pd

import galsim

from astropy import coordinates as coord
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from regions import PolygonSkyRegion

from metacoadd import ExposureBound, ExpBList, PrepCoaddBound
from metacoadd import Exposure, ExpList, CoaddImage
from metacoadd.metacoadd import SimpleCoadd

from .utils import parser
from .utils.header_builder import make_header, make_coadd_wcs
from .utils.catalog import write_catalog_stamp, make_coadd_catalog
from .utils.swarp_wrapper import run_swarp
from .CCDMaker import CCDStampMaker
from .MaskMaker import MaskMaker
from .psf.atmospheric import seeing_distribution


N_CCD = 40


class CoaddStampMaker(object):
    """ """

    def __init__(
        self, stamp_coord, stamp_id, config, gal_catalog, star_catalog
    ):
        # self.coadd_xxx_yyy = coadd_xxx_yyy
        self.stamp_id = stamp_id
        self.config_path = config
        (
            self._atm_config,
            self._opt_config,
            self._file_config,
            self._stamp_config,
            self._simu_config,
        ) = self._load_config(config)
        self._init_coadd_stamp(stamp_coord)
        self._init_catalog(gal_catalog, star_catalog)

        # self._init_output()

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
            config_dict["stamp"],
            config_dict["simu"],
        )

    def _get_coadd_expblist(self, coadd_info):
        exp_bound = galsim.BoundsI(
            xmin=1,
            xmax=self._opt_config["ccd_size"][0],
            ymin=1,
            ymax=self._opt_config["ccd_size"][1],
        )

        self.all_header_info = pd.read_pickle(self._opt_config["header_info"])
        gain = np.load(self._opt_config["gain"])

        expblist = ExpBList()
        for expnum in coadd_info["explist"]:
            header_info = self.all_header_info.loc[expnum]
            header_list = make_header(
                header_info,
                gain,
                self._opt_config["ccd_size"],
                self._opt_config["data_sec"],
            )
            for i in range(40):
                exp = ExposureBound(
                    exp_bound,
                    wcs=galsim.AstropyWCS(header=header_list[i]),
                    meta={
                        "EXPNUM": expnum,
                        "EXTVER": i,
                        "header": header_list[i],
                    },
                )
                expblist.append(exp)

        return expblist

    def _init_coadd_stamp(self, stamp_coord):
        coadd_info_ = pd.read_pickle(self._stamp_config["coadd_info"])
        tile_indice = coadd_info_.index.to_numpy(
            dtype=[("xxx", np.int32), ("yyy", np.int32)]
        )

        # Get CFIS coadd in which the stamp is located.
        coadd_coord = coord.SkyCoord(
            ra=coadd_info_["ra"].values * u.deg,
            dec=coadd_info_["dec"].values * u.deg,
        )
        coadd_wcs = make_coadd_wcs(
            self._stamp_config["coadd_scale"],
            coadd_info_["ra"].values[0],
            coadd_info_["dec"].values[0],
        )
        stamp_coord = coord.SkyCoord(
            ra=stamp_coord[0] * u.deg,
            dec=stamp_coord[1] * u.deg,
        )
        tmp_coadds_mask = stamp_coord.separation(coadd_coord) < 25 * u.arcmin
        tmp_coadds_info = coadd_info_[tmp_coadds_mask]

        for i in range(sum(tmp_coadds_mask)):
            tmp_corner_sky = coord.SkyCoord(
                ra=tmp_coadds_info["corner_ra"].values[i] * u.deg,
                dec=tmp_coadds_info["corner_dec"].values[i] * u.deg,
            )
            w_ = coadd_wcs.copy()
            w_.wcs.crval = np.array(
                [
                    tmp_coadds_info["ra"].values[i],
                    tmp_coadds_info["dec"].values[i],
                ]
            )
            region = PolygonSkyRegion(tmp_corner_sky)
            is_in = region.contains(stamp_coord, w_)
            if is_in:
                ind_coadd = tile_indice[tmp_coadds_mask][i]
                break

        # Gather the exposures that belong to a CFIS coadd
        self.coadd_info = coadd_info_.loc[ind_coadd[0], ind_coadd[1]]
        expblist = self._get_coadd_expblist(self.coadd_info)

        cb = PrepCoaddBound(
            expblist,
            world_coadd_center=galsim.CelestialCoord(
                ra=stamp_coord.ra.deg * galsim.degrees,
                dec=stamp_coord.dec.deg * galsim.degrees,
            ),
            scale=self._stamp_config["coadd_scale"],
            image_coadd_size=self._stamp_config["stamp_size"],
            resize_exposure=True,
            relax_resize=0.00001,
        )
        self.stamp_bounds = cb

    def _init_catalog(self, gal_catalog, star_catalog):
        """ """

        gal_catalog_ap = coord.SkyCoord(
            ra=gal_catalog["ra"] * u.degree, dec=gal_catalog["dec"] * u.degree
        )
        star_catalog_ap = coord.SkyCoord(
            ra=star_catalog["ra"] * u.degree,
            dec=star_catalog["dec"] * u.degree,
        )

        field_center = coord.SkyCoord(
            ra=self.coadd_info["ra"] * u.degree,
            dec=self.coadd_info["dec"] * u.degree,
        )

        m_gal = gal_catalog_ap.separation(field_center) < 0.5 * u.degree
        m_star = star_catalog_ap.separation(field_center) < 0.5 * u.degree

        self.gal_catalog = gal_catalog[m_gal]
        self.star_catalog = star_catalog[m_star]
        self.gal_catalog_ap = gal_catalog_ap[m_gal]
        self.star_catalog_ap = star_catalog_ap[m_star]

    def _init_output(self, objid, g1, g2):
        """ """

        self.output_dir_path = os.path.join(
            self._file_config["output_dir"], str(objid)
        )

        if not os.path.exists(self.output_dir_path):
            os.mkdir(self.output_dir_path)

        self.output_dir_shear_path = []
        for g1_tmp, g2_tmp in zip(g1, g2):
            shear_path_tmp = os.path.join(
                self.output_dir_path, f"shear_{g1_tmp:.2f}_{g2_tmp:.2f}"
            )
            self.output_dir_shear_path.append(shear_path_tmp)
            if not os.path.exists(shear_path_tmp):
                os.mkdir(shear_path_tmp)

    def get_stamp_catalog(self, bound, stamp_bound, do_extended=True):
        """ """

        if self._simu_config["only_center"]:
            c_obj = coord.SkyCoord(
                ra=stamp_bound.world_coadd_center.ra.deg * u.deg,
                dec=stamp_bound.world_coadd_center.dec.deg * u.deg,
            )
            mask_gal = self.gal_catalog_ap.separation(c_obj).arcsec == 0.0
            mask_star = np.zeros_like(self.star_catalog_ap, dtype=bool)
        else:
            wcs_foot = bound.wcs.astropy.copy()

            naxis1 = self._stamp_config["stamp_size"]
            naxis2 = self._stamp_config["stamp_size"]
            wcs_foot.array_shape = np.array([naxis2, naxis1])

            if do_extended:
                ext_size = self._simu_config["bright_obj_stamp_size"]
            else:
                ext_size = 0.0

            wcs_foot_ext = copy.deepcopy(wcs_foot)
            naxis1 = self._stamp_config["stamp_size"] + ext_size
            naxis2 = self._stamp_config["stamp_size"] + ext_size
            wcs_foot_ext.array_shape = np.array([naxis2, naxis1])
            wcs_foot_ext.wcs.crpix += ext_size / 2

            mask_gal = wcs_foot.footprint_contains(self.gal_catalog_ap)
            mask_star = wcs_foot_ext.footprint_contains(self.star_catalog_ap)

        return self.gal_catalog[mask_gal], self.star_catalog[mask_star]

    def stamp_running_func(
        self,
        stamp_bound,
        exp_bound,
        ccd_number,
        g1,
        g2,
        target_seeing,
        seed,
    ):
        """ """

        gal_catalog, star_catalog = self.get_stamp_catalog(
            exp_bound, stamp_bound
        )

        self.gal_catalog_stamp = gal_catalog
        self.star_catalog_stamp = star_catalog

        header_info = self.all_header_info.loc[exp_bound._meta["EXPNUM"]]

        ccd_obj = CCDStampMaker(
            exp_bound,
            self.config_path,
            header_info["background"][ccd_number],
            header_info["EXPTIME"][ccd_number],
            header_info["PHOTZP"][ccd_number],
            target_seeing,
            seed,
        )

        new_header = exp_bound._meta["header"].copy()
        new_header["EXPNUM"] = exp_bound._meta["EXPNUM"]
        new_header.update(
            {
                "CRPIX1": exp_bound.wcs.wcs.wcs.crpix[0],
                "CRPIX2": exp_bound.wcs.wcs.wcs.crpix[1],
                "NAXIS1": exp_bound.image_bounds.xmax
                - exp_bound.image_bounds.xmin
                + 1,
                "NAXIS2": exp_bound.image_bounds.ymax
                - exp_bound.image_bounds.ymin
                + 1,
            }
        )
        new_header["STAMPSEC"] = "[{}:{},{}:{}]".format(
            exp_bound.image_bounds.xmin,
            exp_bound.image_bounds.xmax,
            exp_bound.image_bounds.ymin,
            exp_bound.image_bounds.ymax,
        )
        new_header["EXPTIME"] = header_info["EXPTIME"][ccd_number]
        new_header["SEEING"] = np.round(target_seeing, 3)
        new_header["BKG_LVL"] = int(header_info["background"][ccd_number])

        all_res = []
        for g1_tmp, g2_tmp in zip(g1, g2):
            ccd_obj.reset()
            ccd_imgs, ccd_catalog = ccd_obj.go(
                g1_tmp,
                g2_tmp,
                gal_catalog,
                star_catalog,
            )
            if self._simu_config["do_masking"]:
                maks_maker = MaskMaker(
                    self.config_path,
                    ccd_imgs["sci"],
                    new_header,
                    star_catalog,
                    ccd_imgs["weight"],
                    kind="exp",
                )
                mask = maks_maker.get_mask()
                ccd_imgs["mask"] = mask

            all_res.append(
                {
                    "images": ccd_imgs,
                    "header": new_header,
                    "catalog": ccd_catalog,
                }
            )
        new_header["BKG_RMS"] = np.std(ccd_imgs["bkg"].array)

        return all_res

    def stamp_runner(self, stamp_bound, g1, g2, target_seeing, seed_ori):
        """ """

        all_exposures = {}

        for exp_bound in stamp_bound.expblist:
            expnum = exp_bound._meta["EXPNUM"]
            ccd_number = exp_bound._meta["EXTVER"]
            exp_seed = expnum + seed_ori
            if target_seeing is None:
                seeing_dist = seeing_distribution(
                    self._atm_config["seeing_distribution"],
                    exp_seed,
                )
                exp_target_seeing = seeing_dist.get(1)[0]
            elif isinstance(target_seeing, float):
                exp_target_seeing = target_seeing
            else:
                raise ValueError("target_seeing must be float or None.")

            res = self.stamp_running_func(
                stamp_bound,
                exp_bound,
                ccd_number,
                g1,
                g2,
                exp_target_seeing,
                exp_seed,
            )
            all_exposures[f"{expnum}-{ccd_number}"] = res

        self.all_exp = all_exposures

        return all_exposures

    def coadd_runner(self, all_stamps, stamp_bound, n_shear):
        all_coadds = []
        for i in range(n_shear):
            explist = ExpList()
            all_exp_name = []
            for exp_name in all_stamps.keys():
                img = all_stamps[exp_name][i]["images"]["sci"].array.astype(
                    np.float64
                )
                weight = all_stamps[exp_name][i]["images"]["weight"].array
                header = all_stamps[exp_name][i]["header"]
                exp = Exposure(
                    image=img - header["BKG_LVL"] * weight,
                    weight=weight,
                    wcs=all_stamps[exp_name][i]["images"]["sci"].wcs,
                    meta={"header": header},
                )
                explist.append(exp)
                all_exp_name.append(exp_name)

            if self._simu_config["coadd_method"] == "metacoadd":
                coaddimage = CoaddImage(
                    explist,
                    world_coadd_center=stamp_bound.world_coadd_center,
                    scale=self._stamp_config["coadd_scale"],
                    image_coadd_size=self._stamp_config["stamp_size"],
                    resize_exposure=False,
                )
                sp = SimpleCoadd(coaddimage, do_border=True, border_size=10)
                sp.go()

                coadd_header = fits.Header()
                sp.coaddimage.image.wcs.writeToFitsHeader(
                    coadd_header, sp.coaddimage.image.bounds
                )
                coadd_img = sp.coaddimage.image.array
                coadd_weight = sp.coaddimage.weight.array
                coadd_wcs = coaddimage.coadd_wcs
            elif self._simu_config["coadd_method"] == "swarp":
                coadd_img, coadd_weight, coadd_header = run_swarp(
                    explist,
                    stamp_bound.world_coadd_center,
                    self._stamp_config["stamp_size"],
                    self._stamp_config["coadd_scale"],
                    self._simu_config["swarp_info"],
                )
                astro_wcs = WCS(coadd_header)
                coadd_wcs = galsim.AstropyWCS(wcs=astro_wcs)
                coadd_wcs.astropy = astro_wcs

            coadd_img = galsim.Image(coadd_img, wcs=coadd_wcs)
            coadd_header["MAGZP"] = header["ZPD"]

            for exp_name in all_exp_name:
                coadd_header.add_history(
                    f"input image simu_image-{exp_name}.fits.fz 1 extension(s)"
                )

            out_images = {
                "sci": coadd_img.array,
                "weight": coadd_weight,
            }

            # Get coadd catalog
            if i == 0:
                coadd_cat = make_coadd_catalog(
                    all_stamps,
                    coadd_header,
                    coadd_header["MAGZP"],
                )

            output = {
                "images": out_images,
                "header": coadd_header,
                "catalog": coadd_cat,
            }

            if self._simu_config["do_masking"]:
                mask_maker = MaskMaker(
                    self.config_path,
                    coadd_img,
                    coadd_header,
                    coadd_cat[coadd_cat["type"] == 0],
                    kind="coadd",
                )
                mask = mask_maker.get_mask()
                output["images"]["mask"] = mask.array

            all_coadds.append(output)

        self.all_coadds = all_coadds

        return all_coadds

    def get_mask(self):
        pass

    def write_output(self, all_stamps, coadds, n_shear):
        """ """

        # Write all shear for one exposure stamp
        for i in range(n_shear):
            for expname, res in all_stamps.items():
                # Write images (except PSF)
                primary_hdu = fits.PrimaryHDU()
                hdu_list = fits.HDUList([primary_hdu])
                for img_type, img in res[i]["images"].items():
                    if img_type == "psf":
                        continue
                    img_tmp = img.array
                    # Convert to 16bit (unsigned 16bit)
                    img_tmp = img_tmp.astype(np.uint16)
                    img_tmp[img_tmp < 0] = np.max(img_tmp)

                    hdu_list.append(
                        fits.CompImageHDU(
                            img_tmp,
                            header=res[i]["header"],
                            name=f"{img_type.upper()}",
                        )
                    )
                hdu_list.writeto(
                    os.path.join(
                        self.output_dir_shear_path[i],
                        f"simu_image-{expname}.fits.fz",
                    ),
                    overwrite=True,
                )

                # Write PSF (if any)
                primary_hdu = fits.PrimaryHDU()
                hdu_list = fits.HDUList([primary_hdu])
                if "psf" in res[i]["images"].keys():
                    for j, psf_tmp in enumerate(res[i]["images"]["psf"]):
                        img_tmp = psf_tmp.array
                        psf_header = fits.Header()
                        psf_tmp.wcs.writeToFitsHeader(
                            psf_header, psf_tmp.bounds
                        )
                        hdu_list.append(
                            fits.CompImageHDU(
                                img_tmp,
                                header=psf_header,
                                name=f"{j}",
                            )
                        )
                    hdu_list.writeto(
                        os.path.join(
                            self.output_dir_shear_path[i],
                            f"simu_psf-{expname}.fits.fz",
                        ),
                        overwrite=True,
                    )

                # Write catalogs
                out_cat_name = os.path.join(
                    self.output_dir_shear_path[i],
                    f"simu_cat-{expname}.fits",
                )
                write_catalog_stamp(res[i]["catalog"], out_cat_name)

            # Write coadds
            hdu_list = fits.HDUList()
            hdu_list.append(fits.PrimaryHDU())
            for image_type in coadds[i]["images"]:
                hdu_list.append(
                    fits.CompImageHDU(
                        coadds[i]["images"][image_type],
                        header=coadds[i]["header"],
                        name=image_type.upper(),
                    )
                )
            hdu_list.writeto(
                os.path.join(
                    self.output_dir_shear_path[i],
                    "simu_coadd.fits.fz",
                ),
                overwrite=True,
            )

            # Write coadd catalogs
            out_cat_name = os.path.join(
                self.output_dir_shear_path[i],
                "simu_coadd_cat.fits",
            )
            write_catalog_stamp(coadds[i]["catalog"], out_cat_name)

    def go(self, g1, g2, target_seeing=None, seed=1234):
        """ """

        g1 = np.atleast_1d(g1)
        g2 = np.atleast_1d(g2)

        # for stamp_bound, stamp_id in zip(
        #     self.all_stamp_bounds, self.stamp_ids
        # ):

        # init some output dir
        self._init_output(self.stamp_id, g1, g2)
        # run simulation
        all_stamps = self.stamp_runner(
            self.stamp_bounds, g1, g2, target_seeing, seed
        )
        # make coadd
        coadds = self.coadd_runner(all_stamps, self.stamp_bounds, len(g1))
        # write output
        self.write_output(all_stamps, coadds, len(g1))
