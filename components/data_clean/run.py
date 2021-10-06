#!/usr/bin/env python
"""
Cleaning of data and handling outliers
"""
import os
import wandb
import logging
import argparse
import pandas as pd


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="data_clean")
    run.config.update(args)

    logger.info("Downloading artifact")
    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    
    logger.info("Loading artifact to dataframe")
    df = pd.read_csv(artifact_path)   
    
    logger.info("Cleaning the data") 
    df['last_review'] = pd.to_datetime(df['last_review'])
    idx = df['price'].between(args.min_price, args.max_price)
    df = df[idx].copy()
    
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df = df[idx].copy()
    
    filename = "clean_data"
    df.to_csv(filename, index=False)
    
    logger.info("Creating artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact_name,
        type=args.output_artifact_type,
        description=args.output_artifact_description,
    )
    artifact.add_file(filename)

    logger.info("Logging artifact")
    run.log_artifact(artifact)
    
    os.remove(filename)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cleaning of data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help="Name of the input artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact_name", 
        type=str,
        help="Name of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact_type", 
        type=str,
        help="Type of the output artifact",
        required=True
    )

    parser.add_argument(
        "--output_artifact_description", 
        type=str,
        help="Description of the output artifact",
        required=True
    )

    parser.add_argument(
        "--min_price", 
        type=float,
        help="Minimum number for price",
        required=True
    )

    parser.add_argument(
        "--max_price", 
        type=float,
        help="Maximum number for price",
        required=True
    )


    args = parser.parse_args()

    go(args)
