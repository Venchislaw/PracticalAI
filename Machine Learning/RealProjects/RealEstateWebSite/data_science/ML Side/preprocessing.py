import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


num_feats = []
cat_feats = []
ohe_cat_feats = []


# a
pd.set_option("display.max_columns", 500)


# this function fills feature arrays
# these arrays will be used in preprocessing
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
    # Id feature is just useless
    num_feats.remove("Id")


def preprocess(data_path):
    df = pd.read_csv(data_path)
    feature_cluster(df)
    # print("NoSeWa" in cat_feats)

    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    # selecting numeric features

    num_feats.remove("SalePrice")
    X_train_num = X_train[num_feats]
    X_test_num = X_test[num_feats]

    transform_pipeline = make_pipeline(
        SimpleImputer(strategy="mean"),
        RobustScaler()
    )

    X_train_num = transform_pipeline.fit_transform(X_train_num)
    X_test_num = transform_pipeline.transform(X_test_num)

    X_train_num = pd.DataFrame(X_train_num, columns=num_feats)
    X_test_num = pd.DataFrame(X_test_num, columns=num_feats)


    # categorical data preprocessing

    imputer = SimpleImputer(strategy="most_frequent")
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

    X_train_cat = X_train[cat_feats]
    X_test_cat = X_test[cat_feats]

    # impute values
    X_train_cat = imputer.fit_transform(X_train_cat)
    X_test_cat = imputer.transform(X_test_cat)

    # encode values
    X_train_cat = encoder.fit_transform(X_train_cat)
    X_test_cat = encoder.transform(X_test_cat)

    X_train_cat_ohe = X_train[ohe_cat_feats]
    X_test_cat_ohe = X_test[ohe_cat_feats]

    X_train_cat_ohe = pd.get_dummies(X_train_cat_ohe, dtype="int")
    X_test_cat_ohe = pd.get_dummies(X_test_cat_ohe, dtype="int")


    X_train_cat = pd.DataFrame(X_train_cat, columns=cat_feats)
    X_test_cat = pd.DataFrame(X_test_cat, columns=cat_feats)

    X_train_cat = pd.merge(X_train_cat, X_train_cat_ohe, left_index=True, right_index=True)
    X_test_cat = pd.merge(X_test_cat, X_test_cat_ohe, left_index=True, right_index=True)


    # gotta fix that
    """print(set(X_train_cat.columns).difference(set(ohe_cat_feats)))
    print(ohe_cat_feats)
    print("-" * 60)
    print(X_train_cat[ohe_cat_feats].head())

    X_train_cat = X_train_cat.drop(ohe_cat_feats, axis=1)
    X_test_cat = X_test_cat.drop(ohe_cat_feats, axis=1)"""

    # merge two dataframes
    X_train = pd.merge(X_train_num, X_train_cat, left_index=True, right_index=True)
    X_test = pd.merge(X_test_num, X_test_cat, left_index=True, right_index=True)

    return X_train, X_test, y_train, y_test


X_train, X_test, y_train, y_test = preprocess("/media/venchislav/Говорящий Том/RealProjects/RealEstateWebSite/data_science/Data/train.csv")
print(X_train.head())
