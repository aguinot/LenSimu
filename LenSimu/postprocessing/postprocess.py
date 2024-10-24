import os
from glob import glob
import re

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


def get_stamp(img, x, y, stamp_size):
    fs = FetchStamps(img, int(stamp_size / 2))
    fs.get_pixels(np.round([[x, y]]).astype(int))
    vign = fs.scan()[0].astype(np.float64)

    return vign


class PostProcess:
    def __init__(self, stamp_id, input_dir, vign_size=51):
        self.stamp_id = stamp_id
        self.vign_size = vign_size
        self._init_input(stamp_id, input_dir)

    def _init_input(self, stamp_id, input_dir):
        self.input_dir = os.path.join(input_dir, str(stamp_id))
        self.shear_dirs = glob(os.path.join(self.input_dir, "shear_*"))
        # exp_names = glob("simu_image-*.fits.gz", root_dir=self.shear_dirs[0])
        exp_names = glob(
            os.path.join(self.shear_dirs[0], "simu_image-*.fits.gz")
        )
        exp_names = [os.path.split(exp_name)[1] for exp_name in exp_names]
        self.coadd_cat_name = "simu_coadd_cat.fits"

        exp_nums = []
        for exp_name in exp_names:
            exp_num, ccd_id = re.findall(r"\d+", exp_name)
            exp_nums.append([int(exp_num), int(ccd_id)])
        self.exp_nums = np.array(exp_nums)

    def get_shape(self, img_vign, jacob_list, psfs, psfs_sigma):
        gals = img_vign["SCI"]
        weights = img_vign["WEIGHT"]
        flags = img_vign["MASK"]
        prior = get_prior()

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
            res["obj_id"] = self.stamp_id
            res["n_epoch_model"] = len(gals)
        except Exception:
            res = "fail"

        return res

    def _get_img_info(self, ra, dec, dir_path, coadd_mask=None):
        obj_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

        vign = {"SCI": [], "WEIGHT": [], "MASK": []}
        jacob_list = []
        psfs = []
        psfs_sigma = []
        for exp_num in self.exp_nums:
            exp_name = f"simu_image-{exp_num[0]}-{exp_num[1]}.fits.gz"
            exp_path = os.path.join(dir_path, exp_name)

            h_exp = fits.getheader(exp_path, 1, memmap=False)
            w_exp = WCS(h_exp)

            if not w_exp.footprint_contains(obj_coord):
                continue

            # We invert x, y for the get stamp function
            y, x = w_exp.all_world2pix(ra, dec, 0)

            # Get img
            # First we loop to get the mask
            # This is not ideal but given that we can have or not the HDU we
            # have to do that...
            hdulist = fits.open(exp_path, memmap=False)
            k = 0
            for hdu in hdulist:
                if hdu.name != "MASK":
                    continue
                exp_img = hdu.data
                vign_ = get_stamp(exp_img, x, y, self.vign_size)
                if coadd_mask is not None:
                    vign_ += coadd_mask
                k += 1
            if k == 0:
                vign_ = np.zeros(
                    (
                        self.vign_size,
                        self.vign_size,
                    )
                )
            if len(vign_[vign_ != 0].ravel()) / self.vign_size**2 > 1 / 3:
                continue
            vign["MASK"].append(vign_)

            # This loop need a rework. We only go through HDU 1 and 3...
            # all_names = []
            # for hdu in hdulist:
            #     name = hdu.name
            #     all_names.append(name)
            #     if name in ["PRIMARY", "BKG", "MASK"]:
            #         continue
            #     exp_img = hdu.data
            #     vign_ = get_stamp(exp_img, x, y, self.vign_size)

            #     if name == "SCI":
            #         vign_ -= h_exp["BKG_LVL"]
            #     if name == "WEIGHT":
            #         vign_ *= 1/h_exp["BKG_LVL"]
            #         # vign_ += 1
            #     vign[name].append(vign_)
            
            psf, psf_sig = self._get_psf_info(exp_num, dir_path)
            psfs.append(psf)
            psfs_sigma.append(psf_sig)

            # SCI
            img_ = hdulist[1].data - hdulist[3].data
            vign_ = get_stamp(img_, x, y, self.vign_size)
            noise_sig = np.std(hdulist[3].data)
            # noise_sig = 1e-4
            noise_img = np.random.normal(size=vign_.shape) * noise_sig
            vign_ += noise_img
            vign["SCI"].append(vign_)

            # img_ = hdulist[1].data - hdulist[2].data * hdulist[1].header["BKG_LVL"]
            # vign_ = get_stamp(img_, x, y, self.vign_size)
            # vign["SCI"].append(vign_)
            # img_ = hdulist[1].data - hdulist[2].data * hdulist[1].header["BKG_LVL"]
            # vign_ = get_stamp(img_, x, y, self.vign_size)
            # vign["SCI"].append(vign_)

            # # WEIGHT
            img_ = hdulist[2].data #/ noise_sig**2  # np.var(hdulist[3].data)
            vign_ = get_stamp(img_, x, y, self.vign_size)
            vign["WEIGHT"].append(vign_)

            hdulist.close()

            # Get jacobian
            jacob_tmp = get_jacob(
                w_exp,
                ra,
                dec,
            )
            jacob_list.append(jacob_tmp)

        return vign, jacob_list, psfs, psfs_sigma

    def _get_psf_info(self, exp_num, dir_path):
        # psfs = []
        # psfs_sigma = []

        # for exp_num in self.exp_nums:
        psf_name = f"simu_psf-{exp_num[0]}-{exp_num[1]}.fits.gz"
        exp_cat_name = f"simu_cat-{exp_num[0]}-{exp_num[1]}.fits"
        # Get PSF image
        psf_path = os.path.join(dir_path, psf_name)
        vign_ = fits.getdata(psf_path, 1, memmap=False)

        # Get PSF sigma
        cat_path = os.path.join(dir_path, exp_cat_name)
        cat = fits.getdata(cat_path, 1, memmap=False)

        return vign_, cat["psf_fwhm"][0] / 2.355

    def _write_output(self, outputs):
        for output in outputs:
            output_arr = save_ngmix_data(output[1])

            output_path = os.path.join(output[0], "final_cat_gauss_noise_no_weight.npy")
            np.save(output_path, output_arr)

    def go(self):
        all_outputs = []
        for shear_dir in self.shear_dirs:
            cat = fits.getdata(os.path.join(shear_dir, self.coadd_cat_name))
            img_vign, jacob_list, psfs, psfs_sigma = self._get_img_info(
                cat["ra"][0],
                cat["dec"][0],
                shear_dir,
            )

            res = self.get_shape(img_vign, jacob_list, psfs, psfs_sigma)
            if res == "fail":
                continue

            final_res = compile_results([res])
            all_outputs.append((shear_dir, final_res))

        if len(all_outputs) > 0:
            self._write_output(all_outputs)


class PostProcessDetect(PostProcess):
    def __init__(self, stamp_id, input_dir, vign_size=51):
        super().__init__(stamp_id, input_dir, vign_size)

    def _init_input(self, stamp_id, input_dir):
        self.input_dir = os.path.join(input_dir, str(stamp_id))
        self.shear_dirs = glob(os.path.join(self.input_dir, "shear_*"))
        # exp_names = glob("simu_image-*.fits.gz", root_dir=self.shear_dirs[0])
        exp_names = glob(
            os.path.join(self.shear_dirs[0], "simu_image-*.fits.gz")
        )
        exp_names = [os.path.split(exp_name)[1] for exp_name in exp_names]
        self.coadd_img_name = "simu_coadd.fits.gz"

        exp_nums = []
        for exp_name in exp_names:
            exp_num, ccd_id = re.findall(r"\d+", exp_name)
            exp_nums.append([int(exp_num), int(ccd_id)])
        self.exp_nums = np.array(exp_nums)

    def _get_seg_mask(self, x, y, seg, seg_id):
        vign_ = get_stamp(seg, x, y, self.vign_size)

        vign_[vign_ == seg_id] = 0
        vign_[vign_ > 0] = 1

        return vign_

    def _write_output(self, outputs):
        for output in outputs:
            output_arr = save_detect_ngmix_data(output[1])

            output_path = os.path.join(output[0], "final_cat_gauss_noise_no_weight.npy")
            np.save(output_path, output_arr)

    def go(self):
        all_outputs = []
        for shear_dir in self.shear_dirs:
            coadd_img = fits.open(os.path.join(shear_dir, self.coadd_img_name))
            if coadd_img[3].name == "MASK":
                mask = coadd_img[3].data.astype(np.float64)
            cat, seg = get_cat(
                coadd_img[1].data.astype(np.float64),
                coadd_img[2].data.astype(np.float64),
                mask,
                coadd_img[1].header,
                1.5,
            )
            all_res = []
            good_ind = []
            for ii, cat_ in enumerate(cat):
                if cat_["flag"] > 3:
                    continue
                # if sum(cat["central_flag"]) == 0:
                #     continue
                # else:
                #     ind_central = np.where(cat["central_flag"] == 1)[0]
                coadd_mask = self._get_seg_mask(
                    # cat_["x"],
                    # cat_["y"],
                    cat_["y"],
                    cat_["x"],
                    seg,
                    cat_["number"],
                )
                img_vign, jacob_list, psfs, psfs_sigma = self._get_img_info(
                    cat_["ra"],
                    cat_["dec"],
                    shear_dir,
                    coadd_mask,
                )
                # psfs, psfs_sigma = self._get_psf_info(shear_dir)

                res = self.get_shape(img_vign, jacob_list, psfs, psfs_sigma)
                if res == "fail":
                    continue
                all_res.append(res)
                good_ind.append(ii)

            final_res = compile_results(all_res)
            final_res.update(
                {key: cat[good_ind][key] for key in cat.dtype.names}
            )
            all_outputs.append((shear_dir, final_res))

        if len(all_outputs) > 0:
            self._write_output(all_outputs)
