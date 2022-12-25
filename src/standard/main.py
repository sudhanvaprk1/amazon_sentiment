import mlflow
from properties import get_config
from preprocessing import preprocess_features
from feature_enginnering import engineer_features
from modelling import train_model
import numpy as np
import json

# fetching config
config = get_config()

# setting a new mlflow experiemnt
mlflow.set_experiment(f"experiments/{config['experiment_name']}")
experiment = mlflow.get_experiment_by_name(f"experiments/{config['experiment_name']}")

# getting experiment id
experiment_id = experiment.experiment_id

# starting a new mlflow run
mlflow.start_run(experiment_id=experiment_id, run_name="online")

# fetching mlflow run id
run = mlflow.active_run()
mlflow_run_id = run.info.run_id

# preprocessing the data
if config['preprocess_data']:
    preprocess_features()

# applying feature engineering
if config['engineer_features']:
    x_train_tfidf, x_test_tfidf, y_train, y_test = engineer_features()

# training model as per the tune flag
model, train_predictions, test_predictions, metrics = train_model(tune_model=config['modelling']['tune_flag'])

# logging model artifact
mlflow.sklearn.log_model(model, f"donors_choose_{config['modelling'].get('model_name', 'v1')}")

# saving predictions
with open("data/y_pred_train.npy", "wb") as file:
    np.save(file, train_predictions)

with open("data/y_pred_test.npy", "wb") as file:
    np.save(file, test_predictions)

# logging metrics with mlflow model
for metric, score in metrics.items():
    mlflow.log_metric(metric, score)

result = mlflow.register_model(f"data/artifacts_path/donors_choose_{config['modelling'].get('model_name', 'v1')}", f"donors_choose_{config['modelling'].get('model_name', 'v1')}")
model_version = result.version

model_metadata = {
    "model_name": f"donors_choose_{config['modelling'].get('model_name', 'v1')}",
    "model_version": model_version
}
with open("deploy/model_metadata.json", "w") as outfile:
    json.dump(model_metadata, outfile)

mlflow.end_run()
