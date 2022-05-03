from typing import Dict, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.impute import SimpleImputer

from methods.cross_validation import make_folds

CLASS_NAME = 'class'


def read_data_set(file_path: str) -> Optional[Dict]:
    data = pd.read_csv(file_path)
    data.columns = [*data.columns[:-1], CLASS_NAME]

    X = data.drop(CLASS_NAME, axis=1)
    y = data[CLASS_NAME].values
    y = LabelBinarizer().fit_transform(y).flatten()

    if len(np.unique(y)) > 2:
        # return none if df contains more than 2 classes
        return None

    y_dist = pd.Series(y).value_counts() / len(y)
    if y_dist[1] > 0.5:
        y = 1 - y

    try:
        folds_xy = make_folds(X.values, y)
    except ValueError:
        try:
            enc = OneHotEncoder(handle_unknown='ignore')
            X = enc.fit_transform(X).todense()
            folds_xy = make_folds(X, y)
        except ValueError:
            return None

    return folds_xy


def prepare_fold(fold: Dict):
    train, test = fold.values()

    X_train, y_train = train.values()
    X_test, y_test = test.values()

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    imputer = SimpleImputer(strategy='median')
    imputer.fit(X_train)

    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    return X_train, y_train, X_test, y_test