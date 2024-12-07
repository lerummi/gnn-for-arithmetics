import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer

from arigin.graph.elements import model_to_frame, mask_result_values, join_categorical


class PipelineLabelEncoder(LabelEncoder):

    def fit_transform(self, y, *args, **kwargs):
        y = y.values.flatten()
        return super().fit_transform(y).reshape(-1, 1)

    def transform(self, y, *args, **kwargs):
        y = y.values.flatten()
        return super().transform(y).reshape(-1, 1)


node_features = Pipeline(
    [
        ("dataframe", FunctionTransformer(model_to_frame)),
        ("column_transform", ColumnTransformer(
            [
                (
                    "class_type", 
                    OneHotEncoder(sparse_output=False), ["class", "type"]
                ),
                ("value", "passthrough", ["value"]),
            ],
            remainder="drop"
            )
        ),
        ("fillnan", SimpleImputer(fill_value=0, strategy="constant")),
#        ("poly", PolynomialFeatures(degree=2, interaction_only=True))
    ]
)


node_features_emb = Pipeline(
    [
        ("dataframe", FunctionTransformer(model_to_frame)),
        ("join", FunctionTransformer(
            join_categorical,
            kw_args={"columns": ["class", "type"]}
            )
        ),
        ("column_transform", ColumnTransformer(
            [
                (
                    "class_type", 
                    PipelineLabelEncoder(), ["class_type"]
                )
            ],
            remainder="drop"
            )
        )
    ]
)

node_features_values = Pipeline(
    [
        ("dataframe", FunctionTransformer(model_to_frame)),
        ("column_transform", ColumnTransformer(
            [
                ("1/value", FunctionTransformer(lambda x: 1 / (x + 1e-5)), ["value"]),
                ("value", "passthrough", ["value"]),
            ],
            remainder="drop"
            )
        ),
        ("fillnan", SimpleImputer(fill_value=0, strategy="constant"))
    ]
)

edge_features = Pipeline(
    [
        ("dataframe", FunctionTransformer(model_to_frame)),
        ("column_transform", ColumnTransformer(
            [("class", OneHotEncoder(sparse_output=False), ["class"])],
            remainder="drop"
            )
        )
    ]
)
