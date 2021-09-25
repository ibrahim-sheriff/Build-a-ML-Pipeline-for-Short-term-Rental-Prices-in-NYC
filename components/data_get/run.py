#!/usr/bin/env python
"""
This step is used to load the data from the local directory and upload it as an artifact to W&B
"""
import argparse
import logging
import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="data_get")
    run.config.update(args)

    logger.info("Creating artifact")
    artifact = wandb.Artifact(
        name=args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )
    artifact.add_file(args.input_file)

    logger.info("Logging artifact")
    run.log_artifact(artifact)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This step gets the data")


    parser.add_argument(
        "--input_file", 
        type=str,
        help="Path for the input file",
        required=True
    )

    parser.add_argument(
        "--artifact_name", 
        type=str,
        help="Name for the artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_type", 
        type=str,
        help="Type for the artifact",
        required=True
    )

    parser.add_argument(
        "--artifact_description", 
        type=str,
        help="Description for the artifact",
        required=True
    )


    args = parser.parse_args()

    go(args)
