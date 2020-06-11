import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn_pandas import DataFrameMapper
import numpy as np

from ..._utils.clip_scale_transformer import ClipScaleTransformer
from ..dataset import LogisticRegressionDataset

_XS_MAPPER = DataFrameMapper([
    (["sex"], OneHotEncoder(categories=[["?", "Male", "Female"]], drop='first')),
    (["workclass"], OneHotEncoder(categories=[["?", "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"]], drop='first')),
    (["marital.status"], OneHotEncoder(categories=[["?", "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"]], drop='first')),
    (["relationship"], OneHotEncoder(categories=[["?", "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"]], drop='first')),
    (["education"], OneHotEncoder(categories=[["?", "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school", "Assoc-acdm", "Assoc-voc",
                                               "9th", "7th-8th", "12th", "Masters", "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"]], drop='first')),
    (["occupation"], OneHotEncoder(categories=[["?", "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial", "Prof-specialty",
                                                "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical", "Farming-fishing", "Transport-moving",
                                                "Priv-house-serv", "Protective-serv", "Armed-Forces"]], drop='first')),
    (["native.country"], OneHotEncoder(categories=[["?", "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
                                                    "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
                                                    "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland",
                                                    "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong", "Holand-Netherlands"]], drop='first')),
    (["age"], ClipScaleTransformer(0.0, 90.0, copy=True)),
    (["education.num"], ClipScaleTransformer(0.0, 16.0, copy=True)),
    (["capital.gain"], ClipScaleTransformer(0.0, 99999.0, copy=True)),
    (["capital.loss"], ClipScaleTransformer(0.0, 4356.0, copy=True)),
    (["hours.per.week"], ClipScaleTransformer(0.0, 99, copy=True))])

_YS_MAPPER = DataFrameMapper([
    (["income"], None)
])

ADULT_NUM_COEFS = 99
ADULT_OUTPUT_CLASSES = np.unique(np.array(("<=50K", ">50K"), dtype="<U16"))
FEATURE_NAMES = ("age", "workclass", "fnlwgt", "education", "education.num", "marital.status", "occupation", "relationship", "race", "sex", "capital.gain", "capital.loss", "hours.per.week", "native.country", "income")

def adult_csv_to_dataframe(filename: str) -> pd.DataFrame:
    df = pd.read_csv(filename, names=FEATURE_NAMES)
    return df

def adult_df_to_dataset(df: pd.DataFrame) -> LogisticRegressionDataset:
    xs = _XS_MAPPER.fit_transform(df)
    ys = _YS_MAPPER.fit_transform(df).astype('<U16')
    return LogisticRegressionDataset(xs, ys)
