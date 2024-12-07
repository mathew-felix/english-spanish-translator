import argparse
from source.DatasetDownload import datasetDownload
from source.DatasetPreprocessing import DatasetPreprocessing
from source.Train import Train
from source.Evaluate import evaluate

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run different stages of the ML pipeline.")
    parser.add_argument(
        "--step", 
        type=str, 
        choices=["download", "preprocess", "train", "evaluate"], 
        required=True, 
        help="Step to run: 'download', 'preprocess', 'train', or 'evaluate'"
    )

    args = parser.parse_args()

    # Map arguments to functions
    if args.step == "download":
        print("Starting dataset download...")
        datasetDownload()
    elif args.step == "preprocess":
        print("Starting dataset preprocessing...")
        DatasetPreprocessing()
    elif args.step == "train":
        print("Starting training...")
        Train()
    elif args.step == "evaluate":
        print("Starting evaluation...")
        evaluate()

if __name__ == "__main__":
    main()
