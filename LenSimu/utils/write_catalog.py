from astropy.io import fits


def make_hdu(ccd_cat):

    c_id = fits.Column(
        name="id", array=ccd_cat["id"], format="K"
    )
    c_cat_id = fits.Column(
        name="cat_id", array=ccd_cat["cat_id"], format="K"
    )
    c_ra = fits.Column(
        name="ra", array=ccd_cat["ra"], format="D"
    )
    c_dec = fits.Column(
        name="dec", array=ccd_cat["dec"], format="D"
    )
    c_x = fits.Column(
        name="x", array=ccd_cat["x"], format="D"
    )
    c_y = fits.Column(
        name="y", array=ccd_cat["y"], format="D"
    )
    c_z = fits.Column(
        name="z", array=ccd_cat["z"], format="D"
    )
    c_flux = fits.Column(
        name="flux", array=ccd_cat["flux"], format="D"
    )
    c_mag = fits.Column(
        name="mag", array=ccd_cat["mag"], format="D"
    )
    c_hlr = fits.Column(
        name="hlr", array=ccd_cat["hlr"], format="D"
    )
    c_q = fits.Column(
        name="q", array=ccd_cat["q"], format="D"
    )
    c_beta = fits.Column(
        name="beta", array=ccd_cat["beta"], format="D"
    )
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
    c_psf_g1 = fits.Column(
        name="psf_g1", array=ccd_cat["psf_g1"], format="D"
    )
    c_psf_g2 = fits.Column(
        name="psf_g2", array=ccd_cat["psf_g2"], format="D"
    )
    c_psf_fwhm = fits.Column(
        name="psf_fwhm", array=ccd_cat["psf_fwhm"], format="D"
    )
    c_type = fits.Column(
        name="type", array=ccd_cat["type"], format="I"
    )

    table_hdu = fits.BinTableHDU.from_columns([
        c_id,
        c_cat_id,
        c_ra,
        c_dec,
        c_x,
        c_y,
        c_z,
        c_flux,
        c_mag,
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
    ])

    table_hdu.header['TTYPE1'] = ("id", "simulation ID")
    table_hdu.header['TTYPE2'] = ("cat_id", "input catalogue ID")
    table_hdu.header['TTYPE3'] = ("ra", "right ascension (deg)")
    table_hdu.header['TTYPE4'] = ("dec", "declination (deg)")
    table_hdu.header['TTYPE5'] = ("x", "x image pos (pix)")
    table_hdu.header['TTYPE6'] = ("y", "y image pos (pix)")
    table_hdu.header['TTYPE7'] = ("z", "redshift")
    table_hdu.header['TTYPE8'] = ("flux", "flux r-band")
    table_hdu.header['TTYPE9'] = ("mag", "magnitude r-band")
    table_hdu.header['TTYPE10'] = ("hlr", "half-light-radius")
    table_hdu.header['TTYPE11'] = ("q", "axis ratio b/a")
    table_hdu.header['TTYPE12'] = ("beta", "position angle (deg)")
    table_hdu.header['TTYPE13'] = ("sersic_n", "sersic index")
    table_hdu.header['TTYPE14'] = ("int_g1", "intrinsic ellipticity g1")
    table_hdu.header['TTYPE15'] = ("int_g2", "intrinsic ellipticity g2")
    table_hdu.header['TTYPE16'] = ("shear_g1", "shear_g1")
    table_hdu.header['TTYPE17'] = ("shear_g2", "shear_g2")
    table_hdu.header['TTYPE18'] = ("psf_g1", "measured PSF g1")
    table_hdu.header['TTYPE19'] = ("psf_g2", "measured PSF g2")
    table_hdu.header['TTYPE20'] = ("psf_fwhm", "measured PSF FWHM")
    table_hdu.header['TTYPE21'] = ("type", "object type | 1: gal, 2: star")

    return table_hdu


def write_catalog(full_cat, output_name):

    hdu_list = fits.HDUList()
    hdu_list.append(fits.PrimaryHDU())

    for ccd_n, ccd_cat in enumerate(full_cat):
        ccd_table = make_hdu(ccd_cat)
        ccd_table.name = f"CCD_{ccd_n}"
        hdu_list.append(ccd_table)

    hdu_list.writeto(output_name, overwrite=True)
