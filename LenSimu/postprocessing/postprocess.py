import os
from glob import glob

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy import units as u
from sf_tools.image.stamp import FetchStamps


from .sep_runner import get_cat
from .ngmix_runner import (
    do_ngmix_metacal,
    get_jacob,
    get_prior,
    compile_results,
)
from .make_catalog_runner import save_ngmix_data, save_detect_ngmix_data


class PostProcess:
    def __init__(self, stamp_id, input_dir, vign_size=51):
        self.vign_size = vign_size
        self._init_input(stamp_id, input_dir)

    def _init_input(self, stamp_id, input_dir):
        self.input_dir = os.path.join(input_dir, str(stamp_id))
        self.shear_dirs = glob(os.path.join(self.input_dir, "shear_*"))
        self.exp_names = glob(
            "simu_image-*.fits.fz", root_dir=self.shear_dirs[0]
        )
        self.psf_names = glob(
            "simu_psf-*.fits.fz", root_dir=self.shear_dirs[0]
        )
        self.exp_cat_names = glob(
            "simu_cat-*.fits", root_dir=self.shear_dirs[0]
        )
        self.coadd_cat_name = "simu_coadd_cat.fits"

    def get_shape(self, dir_path):
        cat = fits.getdata(os.path.join(dir_path, self.coadd_cat_name))

        img_vign, jacob_list = self._get_img_info(
            cat["ra"][0], cat["dec"][0], dir_path
        )
        gals = img_vign["SCI"]
        weights = img_vign["WEIGHT"]
        flags = img_vign["MASK"]
        psfs, psfs_sigma = self._get_psf_info(dir_path)
        prior = get_prior()

        cat_path = os.path.join(dir_path, self.exp_cat_names[0])
        cat = fits.getdata(cat_path, 1, memmap=False)
        id_obj = cat["cat_id"][0]

        self.gals = gals
        self.psfs = psfs
        self.psfs_sigma = psfs_sigma
        self.weights = weights
        self.flags = flags
        self.jacob_list = jacob_list
        self.prior = prior

        try:
            res = do_ngmix_metacal(
                gals,
                psfs,
                psfs_sigma,
                weights,
                flags,
                jacob_list,
                prior,
            )
            res["obj_id"] = id_obj
            res["n_epoch_model"] = len(gals)
        except Exception:
            res = "fail"

        return [res]

    def _get_img_info(self, ra, dec, dir_path, coadd_mask=None):
        obj_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

        vign = {"SCI": [], "WEIGHT": [], "MASK": []}
        jacob_list = []
        for exp_name in self.exp_names:
            exp_path = os.path.join(dir_path, exp_name)

            h_exp = fits.getheader(exp_path, 1, memmap=False)
            w_exp = WCS(h_exp)

            if not w_exp.footprint_contains(obj_coord):
                continue

            x, y = w_exp.all_world2pix(ra, dec, 0)

            # Get img
            hdulist = fits.open(exp_path, memmap=False)
            # First we loop to get the mask
            # This is not ideal but given that we can have or not the HDU we
            # have to do that...
            k = 0
            for hdu in hdulist:
                if hdu.name != "MASK":
                    continue
                exp_img = hdu.data
                fs = FetchStamps(exp_img, int(self.vign_size))
                fs.get_pixels(np.round([[x, y]]).astype(int))
                vign_ = fs.scan()[0].astype(np.float64)
                if coadd_mask is not None:
                    vign_ += coadd_mask
            if k == 0:
                vign_ = np.zeros(
                    (
                        self.vign_size,
                        self.vign_size,
                    )
                )
            if len(vign_[vign_ != 0].ravel()) / self.vign_size**2 > 1 / 3.0:
                continue
            vign["MASK"].append(vign_)

            all_names = []
            for hdu in hdulist:
                name = hdu.name
                all_names.append(name)
                if name in ["PRIMARY", "BKG", "MASK"]:
                    continue
                exp_img = hdu.data
                fs = FetchStamps(exp_img, int(self.vign_size))
                fs.get_pixels(np.round([[x, y]]).astype(int))
                vign_ = fs.scan()[0].astype(np.float64)

                if name == "SCI":
                    vign_ -= h_exp["BKG_LVL"]
                vign[name].append(vign_)
            hdulist.close()

            # if "MASK" not in all_names:
            #     vign_ = np.zeros(
            #         (
            #             self.vign_size,
            #             self.vign_size,
            #         )
            #     )
            #     vign["MASK"].append(vign_)
            # if coadd_mask is not None:

            # Get jacobian
            jacob_tmp = get_jacob(
                w_exp,
                ra,
                dec,
            )
            jacob_list.append(jacob_tmp)

        return vign, jacob_list

    def _get_psf_info(self, dir_path):
        psfs = []
        psfs_sigma = []
        for psf_name, exp_cat_name in zip(self.psf_names, self.exp_cat_names):
            # Get PSF image
            psf_path = os.path.join(dir_path, psf_name)
            vign_ = fits.getdata(psf_path, 1, memmap=False)
            psfs.append(vign_)

            # Get PSF sigma
            cat_path = os.path.join(dir_path, exp_cat_name)
            cat = fits.getdata(cat_path, 1, memmap=False)
            psfs_sigma.append(cat["psf_fwhm"][0] / 2.355)

        return psfs, psfs_sigma

    def _write_output(self, outputs):
        for output in outputs:
            output_arr = save_ngmix_data(output[1])

            output_path = os.path.join(output[0], "final_cat.npy")
            np.save(output_path, output_arr)

    def go(self):
        all_outputs = []
        for shear_dir in self.shear_dirs:
            res = self.get_shape(shear_dir)
            if res[0] == "fail":
                continue

            final_res = compile_results(res)
            all_outputs.append((shear_dir, final_res))

        if len(all_outputs) > 0:
            self._write_output(all_outputs)


class PostProcessDetect(PostProcess):
    def __init__(self, stamp_id, input_dir, vign_size=51):
        super().__init__(stamp_id, input_dir, vign_size)

    def _init_input(self, stamp_id, input_dir):
        self.input_dir = os.path.join(input_dir, str(stamp_id))
        self.shear_dirs = glob(os.path.join(self.input_dir, "shear_*"))
        self.exp_names = glob(
            "simu_image-*.fits.fz", root_dir=self.shear_dirs[0]
        )
        self.psf_names = glob(
            "simu_psf-*.fits.fz", root_dir=self.shear_dirs[0]
        )
        self.exp_cat_names = glob(
            "simu_cat-*.fits", root_dir=self.shear_dirs[0]
        )
        self.coadd_img_name = "simu_coadd.fits.fz"

    def get_shape(self, dir_path):
        coadd_img = fits.open(os.path.join(dir_path, self.coadd_img_name))

        cat, seg = get_cat(
            coadd_img[1].data,
            coadd_img[2].data,
            coadd_img[3].data,
            coadd_img[1].header,
            1.5,
        )
        if sum(cat["central_flag"]) == 0:
            return ["fail"], None
        else:
            ind_central = np.where(cat["central_flag"] == 1)[0]

        coadd_mask = self._get_seg_mask(
            cat["x"][ind_central][0],
            cat["y"][ind_central][0],
            seg,
        )

        img_vign, jacob_list = self._get_img_info(
            cat["ra"][ind_central][0],
            cat["dec"][ind_central][0],
            dir_path,
            coadd_mask,
        )
        gals = img_vign["SCI"]
        weights = img_vign["WEIGHT"]
        flags = img_vign["MASK"]
        psfs, psfs_sigma = self._get_psf_info(dir_path)
        prior = get_prior()

        id_obj = cat["number"][ind_central][0]

        self.gals = gals
        self.psfs = psfs
        self.psfs_sigma = psfs_sigma
        self.weights = weights
        self.flags = flags
        self.jacob_list = jacob_list
        self.prior = prior

        try:
            res = do_ngmix_metacal(
                gals,
                psfs,
                psfs_sigma,
                weights,
                flags,
                jacob_list,
                prior,
            )
            res["obj_id"] = id_obj
            res["n_epoch_model"] = len(gals)
        except Exception:
            res = "fail"

        return [res], cat[ind_central]

    def _get_seg_mask(self, x, y, seg):
        fs = FetchStamps(seg, int(self.vign_size))
        fs.get_pixels(np.round([[x, y]]).astype(int))
        vign_ = fs.scan()[0].astype(np.float64)

        vign_[vign_ > 0] = 1

        return vign_

    def _write_output(self, outputs):
        for output in outputs:
            output_arr = save_detect_ngmix_data(output[1])

            output_path = os.path.join(output[0], "final_cat.npy")
            np.save(output_path, output_arr)

    def go(self):
        all_outputs = []
        for shear_dir in self.shear_dirs:
            res, coadd_cat = self.get_shape(shear_dir)
            if res[0] == "fail":
                continue

            final_res = compile_results(res)
            final_res.update(
                {key: coadd_cat[key] for key in coadd_cat.dtype.names}
            )
            all_outputs.append((shear_dir, final_res))

        if len(all_outputs) > 0:
            self._write_output(all_outputs)
