#!/usr/bin/env python
"""
This step used to split the data to train and test
"""
import os
import wandb
import logging
import argparse
import tempfile
import pandas as pd
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="data_split")
    run.config.update(args)

    logging.info(f"Downloading artifact {args.input_data}")
    artifact_local_path = run.use_artifact(args.input_data).file()
    
    df = pd.read_csv(artifact_local_path)
    
    logger.info("Splitting the dataset")
    train_val, test = train_test_split(df, test_size=args.test_size, random_state=args.random_state,
                                       stratify=df[args.stratify] if args.stratify != "none" else None)
    
    for df, split in zip([train_val, test], ["trainval", "test"]):
        logging.info(f"Uploading the {split}_data.csv dataset")
        
        with tempfile.NamedTemporaryFile("w") as fp:            
            df.to_csv(fp.name, index=False)
            
            artifact = wandb.Artifact(
                name=f"{split}_data.csv",
                type=f"{split}_data",
                description=f"{split} split of dataset {args.input_data}",
            )
            
            artifact.add_file(fp.name)
            
            logger.info(f"Logging artifact {split}_data.csv dataset")
            run.log_artifact(artifact)
            
            artifact.wait()




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Splitting the data")


    parser.add_argument(
        "--input_data", 
        type=str,
        help="Input artifact to split (a CSV file)",
        required=True
    )

    parser.add_argument(
        "--test_size", 
        type=float,
        help="Size of the test split. Fraction of the dataset, or number of items",
        required=True
    )

    parser.add_argument(
        "--random_state", 
        type=int,
        help="Seed for the random number generator. Use this for reproducibility",
        required=True,
        default=42,
    )

    parser.add_argument(
        "--stratify", 
        type=str,
        help="Size of the test split. Fraction of the dataset, or number of items",
        required=True,
        default='none'
    )


    args = parser.parse_args()

    go(args)
