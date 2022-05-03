from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedStratifiedKFold

CV_N_FOLDS = 5
CV_N_REPEATS = 2


def make_folds(X, y, scale=True):
    folds = []

    rskf = RepeatedStratifiedKFold(n_splits=CV_N_FOLDS, n_repeats=CV_N_REPEATS, random_state=36851234)

    for train_idx, test_idx in rskf.split(X, y):
        train_set = {'X': X[train_idx].copy(), 'y': y[train_idx].copy()}
        test_set = {'X': X[test_idx].copy(), 'y': y[test_idx].copy()}

        if scale:
            scaler = MinMaxScaler().fit(train_set['X'])
            train_set['X'] = scaler.transform(train_set['X'])
            test_set['X'] = scaler.transform(test_set['X'])

        fold_data = {'train': train_set, 'test': test_set}
        folds.append(fold_data)

    return folds
