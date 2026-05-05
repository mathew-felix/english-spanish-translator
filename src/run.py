import argparse
import logging

from source.DatasetDownload import datasetDownload
from source.DatasetPreprocessing import DatasetPreprocessing
from source.Evaluate import evaluate
from source.Train import Train

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Run different stages of the ML pipeline.",
    )
    parser.add_argument(
        "--step",
        type=str,
        choices=["download", "preprocess", "train", "evaluate"],
        required=True,
        help="Step to run: 'download', 'preprocess', 'train', or 'evaluate'",
    )

    args = parser.parse_args()

    if args.step == "download":
        logger.info("Starting dataset download...")
        datasetDownload()
    elif args.step == "preprocess":
        logger.info("Starting dataset preprocessing...")
        DatasetPreprocessing()
    elif args.step == "train":
        logger.info("Starting training...")
        Train()
    elif args.step == "evaluate":
        logger.info("Starting evaluation...")
        evaluate()


if __name__ == "__main__":
    main()
