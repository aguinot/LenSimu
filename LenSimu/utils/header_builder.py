"""
- Build header from the compiled info.
- Build WCS for coadd images.
"""

import numpy as np

from astropy.io import fits
from astropy.wcs import WCS


HEADER_KEYS = [
    "EXPTIME",
    "CRPIX1",
    "CRPIX2",
    "CRVAL1",
    "CRVAL2",
    "CD1_1",
    "CD1_2",
    "CD2_1",
    "CD2_2",
    "PV1_0",
    "PV1_1",
    "PV1_2",
    "PV1_3",
    "PV1_4",
    "PV1_5",
    "PV1_6",
    "PV1_7",
    "PV1_8",
    "PV1_9",
    "PV1_10",
    "PV2_0",
    "PV2_1",
    "PV2_2",
    "PV2_3",
    "PV2_4",
    "PV2_5",
    "PV2_6",
    "PV2_7",
    "PV2_8",
    "PV2_9",
    "PV2_10",
    "PHOTZP",
]


def make_header(header_info, gain, ccd_size, data_sec):
    header_list = []
    for i in range(40):
        h = fits.Header()
        h["NAXIS"] = 2
        h["NAXIS1"] = ccd_size[0]
        h["NAXIS2"] = ccd_size[1]
        h["EXTVER"] = i
        h["CTYPE1"] = "RA---TPV"
        h["CTYPE2"] = "DEC--TPV"
        h["DETSIZE"] = "[1:{},1:{}]".format(*ccd_size)
        h["DATASEC"] = "[{}:{},{}:{}]".format(*data_sec)
        h["GAIN"] = gain[i]
        for key in HEADER_KEYS:
            h[key] = header_info[key][i]
        header_list.append(h)
        h["ZPD"] = 30.0
        h["ZPR"] = h["PHOTZP"] - h["ZPD"]
        h["FSCALE"] = 10 ** ((h["ZPR"]) / -2.5)

    return header_list


def make_coadd_wcs(coadd_scale, crval1, crval2):
    wcs_coadd = WCS(naxis=2)
    wcs_coadd.wcs.cd = np.array(
        [[-coadd_scale / 3600, 0.0], [0.0, coadd_scale / 3600]]
    )
    wcs_coadd.pixel_shape = (10_000, 10_000)
    wcs_coadd.wcs.crpix = np.array([5_000, 5_000])
    wcs_coadd.wcs.crval = np.array([crval1, crval2])
    wcs_coadd.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return wcs_coadd
