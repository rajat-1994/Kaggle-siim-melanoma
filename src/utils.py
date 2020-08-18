from sklearn.metrics import roc_auc_score
import pandas as pd
import yaml


def metric(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def save_prediction(predictions, path):
    sub_df = pd.read_csv('data/sample_submission.csv')
    sub_df['target'] = predictions
    print(f"Saving at {path}")
    sub_df.to_csv(path, index=False)


def load_config(yaml_filepath):
    with open(yaml_filepath, "r") as stream:
        config = yaml.load(stream)
    return config
