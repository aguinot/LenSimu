from argparse import ArgumentParser
import os
import logging
from datetime import datetime

from LenSimu.postprocessing.postprocess import PostProcess


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "stamp_index",
        help="Tile index to simulate from the tile IDs catalogue.",
        type=int,
    )
    parser.add_argument(
        "-i",
        "--input_dir",
        dest="input_dir",
        help="path to the input directory (output of stamp_runner)",
        type=str,
    )
    parser.add_argument(
        "-v",
        "--vignet_size",
        dest="vignet_size",
        help="size of the vignet for the shape measurement (default: 51)",
        default=51,
        type=int,
    )

    args = parser.parse_args()

    start = args.stamp_index * 1_000
    stop = (args.stamp_index + 1) * 1_000

    logging.basicConfig(
        filename=f"/home/guinot/n17data/simu_LenSimu/logs/log-shape-{args.stamp_index}.txt",  # noqa
        level=logging.INFO,
    )
    logger = logging.getLogger("stamp_runner")

    for i in range(start, stop):
        logger.info("#####")
        logger.info("time: " + datetime.now().strftime("%m/%d/%Y, %H:%M:%S"))
        logger.info(i)
        # if os.path.exists(
        #     f"/n17data/guinot/simu_LenSimu/output_single_obj/{i}/shear_0.00_-0.02/final_cat.npy"  # noqa
        # ):
        #     logger.info("skipping")
        #     continue

        try:
            pp = PostProcess(
                i,
                args.input_dir,
                args.vignet_size,
            )

            pp.go()
        except Exception:
            logger.info("failed", i)
            continue
        if os.path.exists(
            f"/n17data/guinot/simu_LenSimu/output_single_obj/{i}/shear_0.00_-0.02/final_cat.npy"  # noqa
        ):
            logger.info("done")
        else:
            logger.info("failed, should not appear!!")
