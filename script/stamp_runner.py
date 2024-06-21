from argparse import ArgumentParser

import numpy as np

from LenSimu.StampMaker import CoaddStampMaker


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "tile_index",
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
        "-tile",
        "--tile_id_cat",
        dest="tile_cat",
        help="path to tile IDs catalogue",
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

    print("Loading cat...")
    galcat = np.load(args.galcat)
    starcat = np.load(args.starcat)
    print("Cat loaded!!")

    stamp_cat = np.load(args.stamp_cat)
    tile_cat = np.load(args.tile_cat)
    print("Stamp and Tile done!!")

    csm = CoaddStampMaker(
        tile_cat[args.tile_index],
        args.config,
        stamp_cat,
        galcat,
        starcat,
    )
    print("CSM initialized!!")

    print("Going...")
    csm.go([0.02, -0.02, 0.0, 0.0], [0.0, 0.0, 0.02, -0.02])
