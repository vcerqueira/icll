import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

DATA_DIR = 'keel/'
DATA_PATH = os.listdir(DATA_DIR)
if '.DS_Store' in DATA_PATH:
    DATA_PATH.remove('.DS_Store')

MODELS = {
    'RF': RandomForestClassifier,
    'SVM': SVC,
    'LR': LogisticRegression,
}

MODEL_PARAMETERS = {
    'RF': {'n_estimators': 100},
    'SVM': {'probability': True},
    'LR': {},
}
