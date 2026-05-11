import argparse
import logging

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
        from source.DatasetDownload import datasetDownload

        logger.info("Starting dataset download...")
        datasetDownload()
    elif args.step == "preprocess":
        from source.DatasetPreprocessing import DatasetPreprocessing

        logger.info("Starting dataset preprocessing...")
        DatasetPreprocessing()
    elif args.step == "train":
        from source.Train import Train

        logger.info("Starting training...")
        Train()
    elif args.step == "evaluate":
        from source.Evaluate import evaluate

        logger.info("Starting evaluation...")
        evaluate()


if __name__ == "__main__":
    main()
