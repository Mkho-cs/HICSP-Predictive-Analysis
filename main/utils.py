from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_cat_encoder(cols_cat: list)->ColumnTransformer:
    cat_encoder = ColumnTransformer([("cat", OrdinalEncoder(), cols_cat)], remainder='passthrough')
    return cat_encoder

def create_pipeline(pipe: list)->Pipeline:
    pipeline = Pipeline(pipe)
    return pipeline

def pipeline_fit_transform(pipeline: Pipeline, data):
    return pipeline.fit_transform(data)


