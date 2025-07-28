import os
import joblib
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def pipeline_building(numerical_attributes, categorical_attributes):
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    final_pipeline = ColumnTransformer([
        ("num", numerical_pipeline, numerical_attributes),
        ("cat", categorical_pipeline, categorical_attributes)
    ])
    return final_pipeline

if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("housing.csv")

    # Create income categories for stratified sampling
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0, 1.5, 3, 4.5, 6, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        housing = housing.iloc[train_index].drop("income_cat", axis=1)

    housing_labels = housing['median_house_value'].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    numerical_attributes = housing_features.drop('ocean_proximity', axis=1).columns.tolist()
    categorical_attributes = ['ocean_proximity']

    pipeline = pipeline_building(numerical_attributes, categorical_attributes)
    housing_transformed = pipeline.fit_transform(housing_features)

    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    model.fit(housing_transformed, housing_labels)

    scores = cross_val_score(model, housing_transformed, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
    print("Cross-validation RMSE:", np.sqrt(-scores).mean())

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)

else:
    print("Model already trained!")
