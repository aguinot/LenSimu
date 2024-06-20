import tempfile
import os
import subprocess

from astropy.io import fits


def run_swarp(
    explist,
    coadd_center,
    coadd_size,
    coadd_scale,
    swarp_config,
):
    """Run SWARP on a directory of images.

    Parameters:
        coaddimage (str):
        run_dir (str): Directory containing the input images.

    Returns:
        img
        wght
        header
    """

    # Make working dir
    run_dir = swarp_config["run_dir"]
    tmp_dir = tempfile.TemporaryDirectory(dir=run_dir, prefix="swarp-")

    # Make resamp dir
    resamp_dir = os.path.join(tmp_dir.name, "resamp")
    os.makedirs(resamp_dir)

    img_out_path = os.path.join(tmp_dir.name, "coadd.fits")
    wght_out_path = os.path.join(tmp_dir.name, "coadd.weight.fits")

    # Set input images
    swarp_input_path = os.path.join(tmp_dir.name, "swarp_inputs.txt")
    with open(swarp_input_path, "w") as f:
        bkg_val = []
        for exp in explist:
            exp_num = exp._meta["header"]["EXPNUM"]
            ccd_num = exp._meta["header"]["EXTVER"]

            new_img_full_path = os.path.join(
                tmp_dir.name, f"simu_image-{exp_num}-{ccd_num}.fits"
            )
            new_wght_full_path = os.path.join(
                tmp_dir.name, f"simu_image-{exp_num}-{ccd_num}.weight.fits"
            )
            f.write(new_img_full_path + "\n")

            img = exp.image.array
            h = exp._meta["header"]
            hdu = fits.PrimaryHDU(img, h)
            hdu.writeto(new_img_full_path, overwrite=True)

            img = exp.weight.array
            hdu = fits.PrimaryHDU(img, h)
            hdu.writeto(new_wght_full_path, overwrite=True)

            bkg_val.append(str(h["BKG_LVL"]))

    cmd = (
        f"{swarp_config['exec']} -c {swarp_config['config_path']} "
        f"@{swarp_input_path} "
        f"-IMAGEOUT_NAME {img_out_path} -WEIGHTOUT_NAME {wght_out_path} "
        f"-CENTER {coadd_center.ra.deg}"
        f",{coadd_center.dec.deg} "
        f"-IMAGE_SIZE {coadd_size},{coadd_size} "
        f"-PIXEL_SCALE {coadd_scale} "
        f"-BACK_DEFAULT {','.join(bkg_val)} "
        f"-RESAMPLE_DIR {resamp_dir}"
    )

    subprocess.Popen(
        cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).wait()

    # Get output
    img = fits.getdata(img_out_path, 0)
    header = fits.getheader(img_out_path, 0)
    wght = fits.getdata(wght_out_path, 0)

    tmp_dir.cleanup()

    return img, wght, header
