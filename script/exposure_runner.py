from argparse import ArgumentParser

import numpy as np

from LenSimu.ExposureMaker import ExposureMaker


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

    galcat = np.load(args.galcat)
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
        target_seeing=args.target_seeing,
        seed=args.expname + args.seed + 10**6,
    )
