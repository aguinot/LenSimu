import os
from glob import glob
import re

import numpy as np

import h5py

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
    def __init__(
        self,
        stamp_id,
        input_file,
        output_path,
        vign_size=51
    ):
        self.stamp_id = stamp_id
        self.output_path = output_path
        self.vign_size = vign_size
        self._init_input(input_file, stamp_id)

    def _init_input(self, input_file, stamp_id):
        self._f = h5py.File(input_file)
        self.obj_grp = self._f[str(stamp_id)]
        self.grp_shear_names = list(self.obj_grp.keys())

        exp_names = list(
            self.obj_grp[self.grp_shear_names[0]]["exposures"].keys()
        )
        exp_nums = []
        for exp_name in exp_names:
            exp_num, ccd_id = re.findall(r"\d+", exp_name)
            exp_nums.append([int(exp_num), int(ccd_id)])
        self.exp_nums = np.array(exp_nums)

    def close_input(self):
        self._f.close()

    def get_shape(
        self,
        img_vign,
        jacob_list,
        psfs,
        psfs_sigma,
        rng=None,
        n_epoch=None
    ):
        gals = img_vign["SCI"]
        weights = img_vign["WEIGHT"]
        flags = img_vign["MASK"]
        prior = get_prior(rng=rng)

        try:
            res = do_ngmix_metacal(
                gals,
                psfs,
                psfs_sigma,
                weights,
                flags,
                jacob_list,
                prior,
                rng=rng,
                n_epoch=n_epoch,
            )
            res["obj_id"] = self.stamp_id
            res["n_epoch_model"] = len(gals)
        except Exception:
            res = "fail"

        return res

    def _get_img_info(
        self,
        ra,
        dec,
        shear_name,
        rng=None,
        coadd_mask=None,
        no_noise=False,
        good_weight=False,
    ):
        obj_coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

        vign = {"SCI": [], "WEIGHT": [], "MASK": []}
        jacob_list = []
        psfs = []
        psfs_sigma = []
        for exp_num in self.exp_nums:
            exp_name = f"{exp_num[0]}-{exp_num[1]}"
            exp_grp = self.obj_grp[shear_name]["exposures"][exp_name]

            h_exp = exp_grp["SCI"].attrs
            w_exp = WCS(h_exp)

            if not w_exp.footprint_contains(obj_coord):
                continue

            # We invert x, y for the get stamp function
            y, x = w_exp.all_world2pix(ra, dec, 0)

            # Get img
            # First we loop to get the mask
            # This is not ideal but given that we can have or not the HDU we
            # have to do that...
            img_name_list = exp_grp.keys()
            if "MASK" in img_name_list:
                exp_img = np.array(exp_grp["MASK"])
                vign_ = get_stamp(exp_img, x, y, self.vign_size)
            else:
                vign_ = np.zeros(
                    (
                        self.vign_size,
                        self.vign_size,
                    )
                )
            if coadd_mask is not None:
                vign_ += coadd_mask
            if len(vign_[vign_ != 0].ravel()) / self.vign_size**2 > 1 / 3:
                continue
            vign["MASK"].append(vign_)

            psf, psf_sig = self._get_psf_info(exp_grp)
            psfs.append(psf)
            psfs_sigma.append(psf_sig)

            # SCI
            sci_ = np.array(exp_grp["SCI"])
            if no_noise:
                bkg_ = np.array(exp_grp["BKG"])
            else:
                bkg_ = h_exp["BKG_LVL"]
            img_ = sci_ - bkg_
            vign_ = get_stamp(img_, x, y, self.vign_size)

            # Add a bit of noise for the no_noise case
            if no_noise:
                noise_sig = 1e-5
                noise_img = rng.normal(size=vign_.shape) * noise_sig
                vign_ += noise_img
            vign["SCI"].append(vign_)

            # # WEIGHT
            weight_ = np.array(exp_grp["WEIGHT"])
            if good_weight | no_noise:
                if not no_noise:
                    noise_sig = np.std(exp_grp["BKG"])
                weight_ /= noise_sig**2
            vign_ = get_stamp(weight_, x, y, self.vign_size)
            vign["WEIGHT"].append(vign_)

            # Get jacobian
            jacob_tmp = get_jacob(
                w_exp,
                ra,
                dec,
            )
            jacob_list.append(jacob_tmp)

        return vign, jacob_list, psfs, psfs_sigma

    def _get_psf_info(self, exp_grp):

        # Get PSF image
        vign_ = np.array(exp_grp["PSF"])

        # Get PSF sigma
        cat = np.array(exp_grp["CAT"])

        return vign_[0], cat["psf_fwhm"][0] / 2.355

    def _write_output(self, cat_name, outputs):

        main_file = h5py.File(self.output_path, "a")

        str_objid = str(self.stamp_id)
        if str_objid not in main_file:
            obj_grp = main_file.create_group(str_objid)
        else:
            obj_grp = main_file[str_objid]

        for output in outputs:
            output_arr = save_ngmix_data(output[1])

            if output[0] not in obj_grp:
                shear_grp = obj_grp.create_group(output[0])
            else:
                shear_grp = obj_grp[output[0]]
            
            if cat_name in shear_grp:
                del shear_grp[cat_name]
            dset_tmp = shear_grp.create_dataset(
                cat_name,
                data=output_arr,
                compression="gzip",
                compression_opts=9,
            )

        main_file.close()

    def go(self, cat_name, no_noise=False, good_weight=False, single_epoch=False):
        n_epoch = None
        if single_epoch:
            n_epoch = 1
        all_outputs = []
        for shear_name in self.grp_shear_names:
            rng = np.random.RandomState(self.stamp_id)
            cat = np.array(self.obj_grp[shear_name]["coadd"]["CAT"])
            img_vign, jacob_list, psfs, psfs_sigma = self._get_img_info(
                cat["ra"][0],
                cat["dec"][0],
                shear_name,
                rng=rng,
                no_noise=no_noise,
                good_weight=good_weight,
            )

            res = self.get_shape(
                img_vign,
                jacob_list,
                psfs,
                psfs_sigma,
                rng=rng,
                n_epoch=n_epoch,
            )
            if res == "fail":
                continue

            final_res = compile_results([res])
            all_outputs.append((shear_name, final_res))

        if len(all_outputs) > 0:
            self._write_output(cat_name, all_outputs)


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
                    no_noise=no_noise,
                    good_weight=good_weight,
                )

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
