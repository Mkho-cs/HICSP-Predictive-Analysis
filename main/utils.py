from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

"""
Data preprocessing
"""
def create_cat_encoder(cols_cat: list)->ColumnTransformer:
    cat_encoder = ColumnTransformer([("cat", OrdinalEncoder(), cols_cat)], remainder='passthrough')
    return cat_encoder

def create_pipeline(pipe: list)->Pipeline:
    pipeline = Pipeline(pipe)
    return pipeline

def pipeline_fit_transform(pipeline: Pipeline, data):
    return pipeline.fit_transform(data)

def create_minmax_scaler():
    return MinMaxScaler()

"""
Predictive modelling
"""

def train_val_score(model, train: dict, val: dict, output=roc_auc_score, fit=True)->tuple:
    if fit: model.fit(train['x'], train['y'])

