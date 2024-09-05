import copy

import sep
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
# from astropy import units as u

import numpy as np


def get_output_cat(n_obj):
    dtype = [
        ("number", np.int64),
        ("ra", np.float64),
        ("dec", np.float64),
        ("x", np.float64),
        ("y", np.float64),
        ("a", np.float64),
        ("b", np.float64),
        ("elongation", np.float64),
        ("ellipticity", np.float64),
        ("kronrad", np.float64),
        ("flux", np.float64),
        ("flux_err", np.float64),
        ("flux_radius", np.float64),
        ("mag", np.float64),
        ("mag_err", np.float64),
        ("snr", np.float64),
        ("flag", np.int64),
        ("ext_flag", np.int64),
        ("central_flag", np.int64),
    ]

    out = np.array(
        list(map(tuple, np.zeros((len(dtype), n_obj)).T)), dtype=dtype
    )

    return out


def get_cat(img, weight, mask, header, thresh, zp_key="MAGZP"):
    rms = np.zeros_like(weight)
    mask_rms = np.ones_like(weight)
    m = np.where(weight > 0)
    rms[m] = np.sqrt(1 / weight[m])
    mask_rms[m] = 0
    wcs = WCS(header)
    zp = header[zp_key]

    obj, seg = sep.extract(
        img,
        thresh,
        err=rms,
        segmentation_map=True,
        minarea=10,
        filter_type="conv",
        deblend_nthresh=32,
        deblend_cont=0.001,
    )
    n_obj = len(obj)
    seg_id = np.arange(1, n_obj + 1, dtype=np.int32)

    kronrads, krflags = sep.kron_radius(
        img,
        obj["x"],
        obj["y"],
        obj["a"],
        obj["b"],
        obj["theta"],
        6.0,
        seg_id=seg_id,
        segmap=seg,
        mask=mask_rms,
    )
    fluxes = np.ones(n_obj) * -10.0
    fluxerrs = np.ones(n_obj) * -10.0
    flux_rad = np.ones(n_obj) * -10.0
    mag = np.ones(n_obj) * -10.0
    mag_err = np.ones(n_obj) * -10.0
    snr = np.ones(n_obj) * -10.0
    flags = np.ones(n_obj, dtype=np.int64) * 64
    flags_rad = np.ones(n_obj, dtype=np.int64) * 64

    good_flux = (
        (kronrads > 0)
        & (obj["b"] > 0)
        & (obj["a"] >= obj["b"])
        & (obj["theta"] >= -np.pi / 2)
        & (obj["theta"] <= np.pi / 2)
    )
    fluxes[good_flux], fluxerrs[good_flux], flags[good_flux] = sep.sum_ellipse(
        img,
        obj["x"][good_flux],
        obj["y"][good_flux],
        obj["a"][good_flux],
        obj["b"][good_flux],
        obj["theta"][good_flux],
        2.5 * kronrads[good_flux],
        err=rms,
        subpix=1,
        seg_id=seg_id[good_flux],
        segmap=seg,
        mask=mask_rms,
    )

    flux_rad[good_flux], flags_rad[good_flux] = sep.flux_radius(
        img,
        obj["x"][good_flux],
        obj["y"][good_flux],
        6.0 * obj["a"][good_flux],
        0.5,
        normflux=fluxes[good_flux],
        subpix=1,
        seg_id=seg_id[good_flux],
        segmap=seg,
        mask=mask_rms,
    )

    mag[fluxes > 0] = -2.5 * np.log10(fluxes[fluxes > 0]) + zp
    mag_err[fluxes > 0] = (
        2.5 / np.log(10) * (fluxerrs[fluxes > 0] / fluxes[fluxes > 0])
    )
    snr[fluxes > 0] = fluxes[fluxes > 0] / fluxerrs[fluxes > 0]

    ra, dec = wcs.all_pix2world(obj["x"], obj["y"], 0)

    # Build the equivalent to IMAFLAGS_ISO
    # But you only know if the object is flagged or not, you don't get the flag
    ext_flags = np.zeros(n_obj, dtype=int)
    # for i, seg_id_tmp in enumerate(seg_id):
    #     seg_map_tmp = copy.deepcopy(seg)
    #     seg_map_tmp[seg_map_tmp != seg_id_tmp] = 0
    #     check_map = seg_map_tmp + mask
    #     if (check_map > seg_id_tmp).any():
    #         ext_flags[i] = 1

    # Find central obj
    central_flag = np.zeros(n_obj, dtype=int)
    img_center = SkyCoord(
        ra=header["CRVAL1"], dec=header["CRVAL2"], unit="deg"
    )
    obj_coords = SkyCoord(ra=ra, dec=dec, unit="deg")
    sep2d = obj_coords.separation(img_center).deg
    if sep2d.min() / np.abs(header["CD1_1"]) < 5:
        central_flag[sep2d.argmin()] = 1

    out = get_output_cat(n_obj)

    out["number"] = seg_id
    out["ra"] = ra
    out["dec"] = dec
    out["x"] = obj["x"]
    out["y"] = obj["y"]
    out["a"] = obj["a"]
    out["b"] = obj["b"]
    out["elongation"] = obj["a"] / obj["b"]
    out["ellipticity"] = 1. - obj["b"] / obj["a"]
    out["kronrad"] = kronrads
    out["flux"] = fluxes
    out["flux_err"] = fluxerrs
    out["flux_radius"] = flux_rad
    out["mag"] = mag
    out["mag_err"] = mag_err
    out["snr"] = snr
    out["flag"] = obj["flag"] | krflags | flags | flags_rad
    out["ext_flag"] = ext_flags
    out["central_flag"] = central_flag

    return out, seg
