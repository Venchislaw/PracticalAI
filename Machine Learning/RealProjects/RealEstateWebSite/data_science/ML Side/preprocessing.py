import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


num_feats = []
cat_feats = []
ohe_cat_feats = []


def feature_cluster(df):
    for feature in df.columns:
        if df[feature].isnull().sum() >= df.shape[0] * 0.5:
            continue

        if df[feature].nunique() <= 50:
            cat_feats.append(feature)
            if df[feature].nunique() <= 10:
                ohe_cat_feats.append(feature)
        else:
            num_feats.append(feature)


def preprocess_data(data_path):
    df = pd.read_csv(data_path)
    # classify features
    feature_cluster(df)

    df = df[num_feats + cat_feats]
