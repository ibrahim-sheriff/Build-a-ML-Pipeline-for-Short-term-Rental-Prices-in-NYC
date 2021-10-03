import os
import json
import hydra
import mlflow
import tempfile
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_name="config")
def go(config: DictConfig):
    
    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    steps_to_execute = config["main"]["execute_steps"]
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")

    with tempfile.TemporaryDirectory() as tmp_dir:

        if "data_get" in steps_to_execute:
            _ = mlflow.run(
                    os.path.join(root_path, "components", "data_get"),
                    "main",
                    parameters={
                        "input_file": os.path.join(root_path, config["data"]["sample"]),
                        "artifact_name": "raw_data.csv",
                        "artifact_type": "raw_data",
                        "artifact_description": "Input raw dataset from csv file"
                    },
            )
        
        if "data_clean" in steps_to_execute:
            _ = mlflow.run(
                    os.path.join(root_path, "components", "data_clean"),
                    "main",
                    parameters={
                        "input_artifact": "nyc_airbnb/raw_data.csv:latest",
                        "output_artifact_name": "clean_data.csv",
                        "output_artifact_type": "clean_data",
                        "output_artifact_description": "Clean dataset with outliers removed",
                        "min_price": config['etl']['min_price'],
                        "max_price": config['etl']['max_price']
                    },
            )
        
        if "data_check" in steps_to_execute:
            _ = mlflow.run(
                    os.path.join(root_path, "components", "data_check"),
                    "main",
                    parameters={
                        "csv": "nyc_airbnb/clean_data.csv:latest",
                        "ref": "nyc_airbnb/clean_data.csv:reference",
                        "kl_threshold": config['data_check']['kl_threshold'],
                        "min_price": config['etl']['min_price'],
                        "max_price": config['etl']['max_price']
                    },
            )
        
        if "data_split" in steps_to_execute:
            _ = mlflow.run(
                    os.path.join(root_path, "components", "data_split"),
                    "main",
                    parameters={
                        "input_data": "nyc_airbnb/clean_data.csv:latest",
                        "test_size": config['data']['test_size'],
                        "random_state": config['main']['random_state'],
                        "stratify": config['data']['stratify']
                    },
            )
        
        if "train_random_forest" in steps_to_execute:
            
            # NOTE: we need to serialize the random forest configuration into JSON
            rf_config = os.path.abspath("rf_config.json")
            #with open(rf_config, "w+") as fp:
            #    json.dump(dict(config["pipeline"].items()), fp)
            
            with open(rf_config, "w+") as fp:
                fp.write(OmegaConf.to_yaml(config["pipeline"]))
    
            _ = mlflow.run(
                    os.path.join(root_path, "components", "train_random_forest"),
                    "main",
                    parameters={
                        "trainval_artifact": "nyc_airbnb/trainval_data.csv:latest",
                        "val_size": config['data']['val_size'],
                        "random_state": config['main']['random_state'],
                        "stratify": config['data']['stratify'],
                        "rf_config": rf_config,
                        "output_artifact": config['pipeline']['export_artifact']
                    },
            )            
            
        if "test_model" in steps_to_execute:
            _ = mlflow.run(
                    os.path.join(root_path, "components", "test_model"),
                    "main",
                    parameters={
                        "mlflow_model": "nyc_airbnb/" + config['pipeline']['export_artifact'] + ":prod",
                        "test_dataset": "nyc_airbnb/test_data.csv:latest"
                    },
            )   

if __name__ == "__main__":
    go()