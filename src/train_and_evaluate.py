## Load data
## Train and evaluate 
## Save matrix and params

import os
from get_data import read_params
import pandas as pd
import warnings
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import argparse
import json
import joblib

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["split_data"]["train_path"]
    test_data_path = config["split_data"]["test_path"]
    alpha = config["estimators"]["ElasticNet"]["params"]["alpha"]
    l1_ratio = config["estimators"]["ElasticNet"]["params"]["alpha"]
    random_state = config["base"]["random_state"]
    model_dir = config["model_dir"]
    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep = ",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)
    (rms, mae, r2) = eval_metrics(test_y, predicted_qualities)
    #####
    print("ElasticNet (alpha=%f, l1_ratio=%f)" % (alpha, l1_ratio))
    print("Mean Square Error: %s" % rms)
    print("Mean Absolute Error: %s" % mae)
    print("R2_score: %s" % r2)
     ##guardar
    params_file = config["reports"]["params"]
    scores_file = config["reports"]["scores"]

    with open(params_file, "w") as f:
        params = {
            "alpha": alpha,
            "l1_ratio": l1_ratio
        }
        json.dump(params, f, indent=4)
    with open(scores_file, "w") as f:
        scores = {
            "rms": rms,
            "mae": mae,
            "r2": r2
        }
        json.dump(scores, f, indent=4)
    #####    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr, model_path)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default = "params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path = parsed_args.config)