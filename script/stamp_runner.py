from argparse import ArgumentParser
import os
import logging
from datetime import datetime

import numpy as np

from LenSimu.StampMaker import CoaddStampMaker


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
        "stamp_index",
        help="Tile index to simulate from the tile IDs catalogue.",
        type=int,
    )
    parser.add_argument(
        "-c",
        "--config",
        dest="config",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "-gal",
        "--gal_cat",
        dest="galcat",
        help="path to galaxy catalogue",
        type=str,
    )
    parser.add_argument(
        "-star",
        "--star_cat",
        dest="starcat",
        help="path to star catalogue",
        type=str,
    )
    parser.add_argument(
        "-stamp",
        "--stamp_cat",
        dest="stamp_cat",
        help="path to stamp center catalogue",
        type=str,
    )
    parser.add_argument(
        "-s",
        "--seed",
        dest="seed",
        help="seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # galcat = np.load(args.galcat)
    galcat = np.memmap(
        args.galcat,
        dtype=galcat_dtype,
        mode="readonly",
        shape=(size,)
    )
    starcat = np.load(args.starcat)

    stamp_cat = np.load(args.stamp_cat)

    start = args.stamp_index * 1_000
    stop = (args.stamp_index + 1) * 1_000

    logging.basicConfig(
        filename=f"/hildafs/home/aguinot/work/unions_sim/logs/log-single-{args.stamp_index}.txt",  # noqa
        level=logging.INFO,
    )
    logger = logging.getLogger("stamp_runner")

    check_dir = "/hildafs/home/aguinot/work/unions_sim/output/output_stamp_single_new3"

    for i in range(start, stop):
        logger.info("#####")
        logger.info("time: " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        logger.info(i)
        if (
            os.path.exists(
                f"{check_dir}/{i}/shear_0.00_-0.02/simu_coadd.fits.gz"  # noqa
            ) & 
            os.path.exists(
                f"{check_dir}/{i}/shear_0.00_0.02/simu_coadd.fits.gz"  # noqa
            ) &
            os.path.exists(
                f"{check_dir}/{i}/shear_0.02_0.00/simu_coadd.fits.gz"  # noqa
            ) &
            os.path.exists(
                f"{check_dir}/{i}/shear_-0.02_0.00/simu_coadd.fits.gz"  # noqa
            )
        ):
            logger.info("skipping")
            continue

        try:
            csm = CoaddStampMaker(
                stamp_cat[i],
                i,
                args.config,
                galcat,
                starcat,
            )

            csm.go([0.02, -0.02, 0.0, 0.0], [0.0, 0.0, 0.02, -0.02])
        except Exception:
            logger.info("failed", i)
            continue
        if os.path.exists(
            f"{check_dir}/{i}/shear_0.00_-0.02/simu_coadd.fits.gz"  # noqa
        ):
            logger.info("done")
        else:
            logger.info("failed, should not appear!!")
