#!/usr/bin/env python
"""
This script builds a pipeline and trains a Random Forest model
"""
import os
import yaml
import wandb
import mlflow
import shutil
import logging
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def delta_date_feature(dates):
    """
    Given a 2d array containing dates (in any format recognized by pd.to_datetime), it returns the delta in days
    between each date and the most recent date in its column
    """
    date_sanitized = pd.DataFrame(dates).apply(pd.to_datetime)
    return date_sanitized.apply(lambda d: (d.max() -d).dt.days, axis=0).to_numpy()


def get_inference_pipeline(rf_config):
    
    # categorical feature preprocessor
    categorical_preproc = make_pipeline(
        SimpleImputer(strategy='most_frequent'),
        OneHotEncoder()
    )
    
    # NOTE: we do not need to impute room_type because the type of the room
    # is mandatory on the websites, so missing values are not possible in production
    # (nor during training). That is not true for neighbourhood_group
    
    # ordinal categorical feature preprocessor
    ordinal_categ_preproc = OrdinalEncoder()
    
    # numerical features preprocessor
    numerical_preproc = SimpleImputer(strategy='constant', fill_value=0)
    
    # date feature preprocessor
    date_preproc = make_pipeline(
        SimpleImputer(strategy='constant', fill_value='2010-01-01'),
        FunctionTransformer(delta_date_feature, validate=False, check_inverse=False)
    )
    
    # text feature preprocessor
    reshape_to_1d = FunctionTransformer(np.reshape, kw_args={"newshape": -1})
    text_preproc = make_pipeline(
        SimpleImputer(strategy='constant', fill_value=''),
        reshape_to_1d,
        TfidfVectorizer(
            max_features=rf_config['tfidf']['max_features'],
            binary=False,
            stop_words='english'
        )
    )
    
    # all features preprocessor using column transformer
    features_preprocessor = ColumnTransformer([
            ('categorical', categorical_preproc, rf_config['features']['categorical']),
            ('ordinal_categ', ordinal_categ_preproc, rf_config['features']['ordinal_categ']),
            ('numerical', numerical_preproc, rf_config['features']['numerical']),
            ('date', date_preproc, rf_config['features']['date']),
            ('text', text_preproc, rf_config['features']['text'])
        ],
        remainder='drop'                                   
    )
    
    # random forest model
    rf_model = RandomForestRegressor(**rf_config['model']['random_forest'])
    
    # inference pipeline
    pipe = Pipeline([
        ('features_preprocessor', features_preprocessor),
        ('random_forest_model', rf_model)
    ])
    
    # Get a list of the columns we used
    processed_features = list(itertools.chain.from_iterable([x[2] for x in features_preprocessor.transformers]))
    
    return pipe, processed_features


def plot_feature_importance(pipe, feat_names):
    # We collect the feature importance for all non-nlp features first
    feat_imp = pipe["random_forest_model"].feature_importances_[: len(feat_names)-1]
    # For the NLP feature we sum across all the TF-IDF dimensions into a global
    # NLP importance
    nlp_importance = sum(pipe["random_forest_model"].feature_importances_[len(feat_names) - 1:])
    feat_imp = np.append(feat_imp, nlp_importance)
    fig_feat_imp, sub_feat_imp = plt.subplots(figsize=(10, 10))
    # idx = np.argsort(feat_imp)[::-1]
    sub_feat_imp.bar(range(feat_imp.shape[0]), feat_imp, color="r", align="center")
    _ = sub_feat_imp.set_xticks(range(feat_imp.shape[0]))
    _ = sub_feat_imp.set_xticklabels(np.array(feat_names), rotation=90)
    fig_feat_imp.tight_layout()
    return fig_feat_imp
    

def go(args):

    run = wandb.init(job_type="train_random_forest")
    run.config.update(args)

    # get the random forest configuration file and update W&B 
    with open(args.rf_config) as fp:
        rf_config = yaml.safe_load(fp)
    run.config.update(rf_config)
    
    rf_config['random_state'] = args.random_state
    
    
    # load the dataset and split to train and val
    logger.info("Loading the dataset")
    data_path = run.use_artifact(args.trainval_artifact).file()
    X_df = pd.read_csv(data_path)
    y_df = X_df.pop('price')
    
    logger.info("Spliting the dataset to train and validation")
    X_train, X_val, y_train, y_val = train_test_split(
        X_df, y_df, test_size=args.val_size, random_state=args.random_state,
        stratify=X_df[args.stratify] if args.stratify != "none" else None
    )
    
    logger.info("Creating random forest inference pipeline")
    rf_pipeline, processed_features = get_inference_pipeline(rf_config)
    
    logger.info("Training random forest model")
    rf_pipeline.fit(X_train, y_train)
    
    logger.info("Predicting validation data")
    y_pred = rf_pipeline.predict(X_val)
    
    logger.info("Scoring")
    r_squared = r2_score(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    
    logger.info(f"Validation data R squared: {r_squared}")
    logger.info(f"Validation data mean absolute error: {mae}")
    
    logger.info("Exporting inference pipeline")
    signature = mlflow.models.infer_signature(X_val[processed_features], y_pred)

    if os.path.exists("models/random_forest"):
        shutil.rmtree("models/random_forest")

    mlflow.sklearn.save_model(
        rf_pipeline,
        "models/random_forest",
        signature=signature,
        input_example=X_val.iloc[:5]
    )

    logger.info("Uploading model artifact")
    artifact = wandb.Artifact(
        name=args.output_artifact,
        type="model_export", 
        description="Random forest inference pipelie"
    )
    
    artifact.add_dir("models/random_forest")
    run.log_artifact(artifact)
    
    logger.info("Create feature importance model")
    fig_feat_imp = plot_feature_importance(rf_pipeline, processed_features)
    
    logger.info("Logging metrics and losses")
    run.summary['r2'] = r_squared
    run.summary['mae'] = mae
    
    # Upload to W&B the feature importance visualization
    run.log(
        {
          "feature_importance": wandb.Image(fig_feat_imp),
        }
    )
    
    
     
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training a random forest model")

    parser.add_argument(
        "--trainval_artifact",
        type=str,
        help="Artifact containing the training dataset. It will be split into train and validation"
    )

    parser.add_argument(
        "--val_size",
        type=float,
        help="Size of the validation split. Fraction of the dataset, or number of items",
    )

    parser.add_argument(
        "--random_state",
        type=int,
        help="Seed for random number generator",
        default=42,
        required=False,
    )

    parser.add_argument(
        "--stratify",
        type=str,
        help="Column to use for stratification",
        default="none",
        required=False,
    )

    parser.add_argument(
        "--rf_config",
        help="Random forest configuration. A JSON dict that will be passed to the "
        "scikit-learn constructor for RandomForestRegressor.",
        default="{}",
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Name for the output serialized model",
        required=True,
    )

    args = parser.parse_args()

    go(args)