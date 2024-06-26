import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

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

    X_train, X_test, y_train, y_test = train_test_split(df, train_size=0.8)

    X_train = pd.DataFrame(X_train, df.columns)
    X_test = pd.DataFrame(X_test, df.columns)

    numeric_df = df[num_feats]
    categoric_df = df[cat_feats]

    num_pipeline = make_pipeline(
        SimpleImputer("mean"),
        RobustScaler()
    )

    cat_pipeline = make_pipeline(
        SimpleImputer("most_frequent"),
        LabelEncoder()
    )

    X_train_num = X_train[num_feats]
    X_test_num = X_test[num_feats]

    X_train_cat = X_train[cat_feats]
    X_test_cat = X_test[cat_feats]

    # numero
    X_train_num = pd.DataFrame(
        num_pipeline.fit_transform(X_train_num),
        columns=num_feats
    )

    X_test_num = pd.DataFrame(
        num_pipeline.transform(X_test_num),
        columns=num_feats
    )

    # categorical
    X_train_cat = pd.DataFrame(
        cat_pipeline.fit_transform(X_train_cat),
        columns=num_feats
    )

    X_test_cat = pd.DataFrame(
        cat_pipeline.transform(X_test_cat),
        columns=num_feats
    )

    X_train_ohe_cat = pd.get_dummies(X_train_cat[ohe_cat_feats])
    X_test_ohe_cat = pd.get_dummies(X_test_cat[ohe_cat_feats])

    X_train_cat = pd.merge(X_train_cat, X_train_ohe_cat, left_index=True, right_index=True)
    X_test_cat = pd.merge(X_test_cat, X_test_ohe_cat, left_index=True, right_index=True)

    X_train_cat = X_train.drop(ohe_cat_feats, axis=1)
    X_test_cat = X_test.drop(ohe_cat_feats, axis=1)

    X_train = pd.merge(X_train_num, X_train_cat, left_index=True, right_index=True)
    X_test = pd.merge(X_test_num, X_test_cat, left_index=True, right_index=True)

    return X_train, X_test
