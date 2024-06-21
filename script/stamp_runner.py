from argparse import ArgumentParser

import numpy as np

from LenSimu.StampMaker import CoaddStampMaker


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

    galcat = np.load(args.galcat)
    starcat = np.load(args.starcat)

    stamp_cat = np.load(args.stamp_cat)

    csm = CoaddStampMaker(
        stamp_cat[args.stamp_index],
        args.stamp_index,
        args.config,
        galcat,
        starcat,
    )

    csm.go([0.02, -0.02, 0.0, 0.0], [0.0, 0.0, 0.02, -0.02])
