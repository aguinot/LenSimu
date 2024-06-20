import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.wcs import WCS


def make_hdu(ccd_cat):
    c_id = fits.Column(name="id", array=ccd_cat["id"], format="K")
    c_cat_id = fits.Column(name="cat_id", array=ccd_cat["cat_id"], format="K")
    c_ra = fits.Column(name="ra", array=ccd_cat["ra"], format="D")
    c_dec = fits.Column(name="dec", array=ccd_cat["dec"], format="D")
    c_x = fits.Column(name="x", array=ccd_cat["x"], format="D")
    c_y = fits.Column(name="y", array=ccd_cat["y"], format="D")
    c_z = fits.Column(name="z", array=ccd_cat["z"], format="D")
    c_flux = fits.Column(name="flux", array=ccd_cat["flux"], format="D")
    c_r_mag = fits.Column(name="r_mag", array=ccd_cat["r_mag"], format="D")
    c_J_mag = fits.Column(name="J_mag", array=ccd_cat["J_mag"], format="D")
    c_hlr = fits.Column(name="hlr", array=ccd_cat["hlr"], format="D")
    c_q = fits.Column(name="q", array=ccd_cat["q"], format="D")
    c_beta = fits.Column(name="beta", array=ccd_cat["beta"], format="D")
    c_sersic_n = fits.Column(
        name="sersic_n", array=ccd_cat["sersic_n"], format="D"
    )
    c_int_g1 = fits.Column(
        name="int_g1", array=ccd_cat["intrinsic_g1"], format="D"
    )
    c_int_g2 = fits.Column(
        name="int_g2", array=ccd_cat["intrinsic_g2"], format="D"
    )
    c_shear_g1 = fits.Column(
        name="shear_g1", array=ccd_cat["shear_g1"], format="D"
    )
    c_shear_g2 = fits.Column(
        name="shear_g2", array=ccd_cat["shear_g2"], format="D"
    )
    c_psf_g1 = fits.Column(name="psf_g1", array=ccd_cat["psf_g1"], format="D")
    c_psf_g2 = fits.Column(name="psf_g2", array=ccd_cat["psf_g2"], format="D")
    c_psf_fwhm = fits.Column(
        name="psf_fwhm", array=ccd_cat["psf_fwhm"], format="D"
    )
    c_type = fits.Column(name="type", array=ccd_cat["type"], format="I")

    table_hdu = fits.BinTableHDU.from_columns(
        [
            c_id,
            c_cat_id,
            c_ra,
            c_dec,
            c_x,
            c_y,
            c_z,
            c_flux,
            c_r_mag,
            c_J_mag,
            c_hlr,
            c_q,
            c_beta,
            c_sersic_n,
            c_int_g1,
            c_int_g2,
            c_shear_g1,
            c_shear_g2,
            c_psf_g1,
            c_psf_g2,
            c_psf_fwhm,
            c_type,
        ]
    )

    table_hdu.header["TTYPE1"] = ("id", "simulation ID")
    table_hdu.header["TTYPE2"] = ("cat_id", "input catalogue ID")
    table_hdu.header["TTYPE3"] = ("ra", "right ascension (deg)")
    table_hdu.header["TTYPE4"] = ("dec", "declination (deg)")
    table_hdu.header["TTYPE5"] = ("x", "x image pos (pix)")
    table_hdu.header["TTYPE6"] = ("y", "y image pos (pix)")
    table_hdu.header["TTYPE7"] = ("z", "redshift")
    table_hdu.header["TTYPE8"] = ("flux", "flux r-band")
    table_hdu.header["TTYPE9"] = ("r_mag", "magnitude r-band")
    table_hdu.header["TTYPE10"] = ("J_mag", "magnitude J-band")
    table_hdu.header["TTYPE11"] = ("hlr", "half-light-radius")
    table_hdu.header["TTYPE12"] = ("q", "axis ratio b/a")
    table_hdu.header["TTYPE13"] = ("beta", "position angle (deg)")
    table_hdu.header["TTYPE14"] = ("sersic_n", "sersic index")
    table_hdu.header["TTYPE15"] = ("int_g1", "intrinsic ellipticity g1")
    table_hdu.header["TTYPE16"] = ("int_g2", "intrinsic ellipticity g2")
    table_hdu.header["TTYPE17"] = ("shear_g1", "shear_g1")
    table_hdu.header["TTYPE18"] = ("shear_g2", "shear_g2")
    table_hdu.header["TTYPE19"] = ("psf_g1", "measured PSF g1")
    table_hdu.header["TTYPE20"] = ("psf_g2", "measured PSF g2")
    table_hdu.header["TTYPE21"] = ("psf_fwhm", "measured PSF FWHM")
    table_hdu.header["TTYPE22"] = ("type", "object type | 1: gal, 2: star")

    return table_hdu


def write_catalog(full_cat, output_name):
    hdu_list = fits.HDUList()
    hdu_list.append(fits.PrimaryHDU())

    for ccd_n, ccd_cat in enumerate(full_cat):
        ccd_table = make_hdu(ccd_cat)
        ccd_table.name = f"CCD_{ccd_n}"
        hdu_list.append(ccd_table)

    hdu_list.writeto(output_name, overwrite=True)


def write_catalog_stamp(cat, output_name):
    hdu_list = fits.HDUList()
    hdu_list.append(fits.PrimaryHDU())

    ccd_table = make_hdu(cat)
    ccd_table.name = "TABLE"
    hdu_list.append(ccd_table)

    hdu_list.writeto(output_name, overwrite=True)


def make_coadd_catalog(all_exp, coadd_header, coadd_zp):
    # Merge catalogs
    all_cat = []
    for key, exp in all_exp.items():
        all_cat.append(exp[0]["catalog"])
    all_cat = np.concatenate(all_cat)

    # Separate gal and stars and remove duplicates
    gal_cat = all_cat[all_cat["type"] == 1]
    _, ind_unique = np.unique(gal_cat["cat_id"], return_index=True)
    gal_cat = gal_cat[ind_unique]

    star_cat = all_cat[all_cat["type"] == 0]
    _, ind_unique = np.unique(star_cat["cat_id"], return_index=True)
    star_cat = star_cat[ind_unique]

    cat = np.concatenate([gal_cat, star_cat])

    # Only keep objects in the coadd image
    coadd_wcs = WCS(coadd_header)
    obj_coord = SkyCoord(ra=cat["ra"] * u.deg, dec=cat["dec"] * u.deg)
    mask_in = coadd_wcs.footprint_contains(obj_coord)
    cat = cat[mask_in]

    # Adapt columns to coadd
    cat["id"] = np.arange(len(cat))
    x, y = coadd_wcs.all_world2pix(cat["ra"], cat["dec"], 0)
    cat["x"] = x
    cat["y"] = y
    cat["psf_g1"] = -10
    cat["psf_g2"] = -10
    cat["psf_fwhm"] = -10
    cat["flux"] = 10 ** (-0.4 * (cat["r_mag"] - coadd_zp))

    return cat
