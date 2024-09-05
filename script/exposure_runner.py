from argparse import ArgumentParser

import numpy as np

from LenSimu.ExposureMaker import ExposureMaker


galcat_dtype = np.dtype([
    ('index', '<i8'),
    ('zobs', '<f8'),
    ('ra', '<f8'),
    ('dec', '<f8'),
    ('Re_arcsec', '<f8'),
    ('BA', '<f8'),
    ('shape/sersic_n', '<f8'),
    ('PA_random', '<f8'),
    ('u_SDSS_apparent_corr', '<f8'),
    ('g_SDSS_apparent_corr', '<f8'),
    ('r_SDSS_apparent_corr', '<f8'),
    ('i_SDSS_apparent_corr', '<f8'),
    ('z_SDSS_apparent_corr', '<f8'),
    ('Y_VISTA_apparent_corr', '<f8'),
    ('J_VISTA_apparent_corr', '<f8'),
    ('H_VISTA_apparent_corr', '<f8'),
    ('K_VISTA_apparent_corr', '<f8'),
    ('Dmag_corr', '<f8'),
])
size = 45487978


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "expname",
        # dest="expname",
        help="exposure name to simulate",
        type=int,
    )
    parser.add_argument(
        "-c", "--config",
        dest="config",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "-gal", "--gal_cat",
        dest="galcat",
        help="path to galaxy catalogue",
        type=str,
    )
    parser.add_argument(
        "-star", "--star_cat",
        dest="starcat",
        help="path to star catalogue",
        type=str,
    )
    parser.add_argument(
        "-g1",
        dest="g1",
        help="shear g1 to apply",
        type=float,
        default=0.,
    )
    parser.add_argument(
        "-g2",
        dest="g2",
        help="shear g2 to apply",
        type=float,
        default=0.,
    )
    parser.add_argument(
        "-ccd",
        dest="ccd_num",
        help="CCD to simulate. If None make all. [Default: None]",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-s", "--seed",
        dest="seed",
        help="seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "-seeing", "--target_seeing",
        dest="target_seeing",
        help="seeing to use for the exposure",
        type=float,
        default=None,
    )

    args = parser.parse_args()

    galcat = np.memmap(
        args.galcat,
        dtype=galcat_dtype,
        mode="readonly",
        shape=(size,)
    )
    # galcat = np.load(args.galcat)
    starcat = np.load(args.starcat)

    ExpMaker = ExposureMaker(
        args.expname,
        args.config,
        galcat,
        starcat,
    )

    ExpMaker.go(
        args.g1,
        args.g2,
        ccd_num=args.ccd_num,
        target_seeing=args.target_seeing,
        seed=args.seed,
    )
