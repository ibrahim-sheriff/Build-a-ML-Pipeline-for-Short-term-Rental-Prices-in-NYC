import os
import hydra
import mlflow
import tempfile
from omegaconf import DictConfig


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
                        "artifact_name": "raw_data",
                        "artifact_type": "data",
                        "artifact_description": "Input raw dataset from csv file"
                    },
            )
        
        if "data_eda" in steps_to_execute:
            # TODO
            pass
        
        if "data_preprocess" in steps_to_execute:
            # TODO
            pass
        
        if "data_check" in steps_to_execute:
            # TODO
            pass
        
        if "data_split" in steps_to_execute:
            # TODO
            pass
        
        if "train_model" in steps_to_execute:
            # TODO
            pass
        
        if "test_model" in steps_to_execute:
            # TODO
            pass

if __name__ == "__main__":
    go()