from argparse import ArgumentParser
import os

import numpy as np

from LenSimu.StampMaker import CoaddStampMaker


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "expname",
        # dest="expname",
        help="exposure name to simulate",
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
        "-tile",
        "--tile_cat",
        dest="tile_cat",
        help="path to tile indices catalogue",
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

    galcat = np.load(args.galcat)
    starcat = np.load(args.starcat)

    stamp_cat = np.load(args.stamp_cat)
    tile_cat = np.load(args.tile_cat)

    tile_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    csm = CoaddStampMaker(
        tile_cat[tile_id],
        args.config,
        stamp_cat,
        galcat,
        starcat,
    )

    csm.go([0.02, -0.02, 0.0, 0.0], [0.0, 0.0, 0.02, -0.02])