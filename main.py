import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import NearMiss, OneSidedSelection, RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier

from methods.icll import ICLL, NoGreyZoneError
from methods.cure import CureModel
from methods.benchmarks import VanillaClassifier, ResampledClassifier
from methods.data_reader import read_data_set, prepare_fold
from methods.metric import AUC

from config import DATA_PATH, MODELS, MODEL_PARAMETERS, DATA_DIR

method = 'RF'


def main():
    scores_across_datasets = []
    no_grey_zone = {}
    for file in DATA_PATH:
        print(file)
        file_path = f'{DATA_DIR}{file}/{file}.csv'

        folds = read_data_set(file_path)

        if folds is None:
            continue

        scores = []
        for data_fold in folds:
            X_train, y_train, X_test, y_test = prepare_fold(data_fold)

            print('Training vanilla classifier (no resampling)')
            baseline = VanillaClassifier(model=MODELS[method](**MODEL_PARAMETERS[method]))
            baseline.fit(X_train, y_train)
            baseline_prob = baseline.predict_proba(X_test)

            print('Training svm (no resampling)')
            baseline_svm = VanillaClassifier(model=MODELS['SVM'](**MODEL_PARAMETERS['SVM']))
            baseline_svm.fit(X_train, y_train)
            baseline_svm_prob = baseline_svm.predict_proba(X_test)

            print('Training lr (no resampling)')
            baseline_lr = VanillaClassifier(model=MODELS['LR'](**MODEL_PARAMETERS['LR']))
            baseline_lr.fit(X_train, y_train)
            baseline_lr_prob = baseline_lr.predict_proba(X_test)

            print('Training balanced rf')
            balanced_rf = BalancedRandomForestClassifier(n_estimators=MODEL_PARAMETERS['RF']['n_estimators'])
            balanced_rf.fit(X_train, y_train)
            balancedrf_prob = balanced_rf.predict_proba(X_test)
            balancedrf_prob = np.array([x[1] for x in balancedrf_prob])

            print('Training baseline SMOTE')
            try:
                clf_smote = ResampledClassifier(model=MODELS[method](**MODEL_PARAMETERS[method]),
                                                resampling_model=SMOTE())
                clf_smote.fit(X_train, y_train)
                clf_smote_prob = clf_smote.predict_proba(X_test)
            except (ValueError, RuntimeError) as e:
                clf_smote_prob = np.repeat(np.nan, len(baseline_prob))

            print('Training baseline NM')
            clf_nearmiss = ResampledClassifier(model=MODELS[method](**MODEL_PARAMETERS[method]),
                                               resampling_model=NearMiss())
            clf_nearmiss.fit(X_train, y_train)
            clf_nearmiss_prob = clf_nearmiss.predict_proba(X_test)

            print('Training baseline RO')
            clf_ro = ResampledClassifier(model=MODELS[method](**MODEL_PARAMETERS[method]),
                                         resampling_model=RandomOverSampler())
            clf_ro.fit(X_train, y_train)
            clf_ro_prob = clf_ro.predict_proba(X_test)

            print('Training baseline RU')
            clf_ru = ResampledClassifier(model=MODELS[method](**MODEL_PARAMETERS[method]),
                                         resampling_model=RandomUnderSampler())
            clf_ru.fit(X_train, y_train)
            clf_ru_prob = clf_ru.predict_proba(X_test)

            print('Training baseline OSS')
            clf_oss = ResampledClassifier(model=MODELS[method](**MODEL_PARAMETERS[method]),
                                          resampling_model=OneSidedSelection())
            clf_oss.fit(X_train, y_train)
            clf_oss_prob = clf_oss.predict_proba(X_test)

            print('Training baseline AD')
            try:
                clf_ada = ResampledClassifier(model=MODELS[method](**MODEL_PARAMETERS[method]),
                                              resampling_model=ADASYN())
                clf_ada.fit(X_train, y_train)
                clf_ada_prob = clf_ada.predict_proba(X_test)
            except (ValueError, RuntimeError) as e:
                clf_ada_prob = np.repeat(np.nan, len(baseline_prob))

            print('Training CURE')
            cure = CureModel(model=MODELS[method](**MODEL_PARAMETERS[method]))
            cure.fit(X_train, y_train)
            cure_prob = cure.predict_proba(X_test)

            print('Training ICLL')
            icll = ICLL(apply_resample_l1=False,
                        apply_resample_l2=False,
                        model_l1=MODELS[method](**MODEL_PARAMETERS[method]),
                        model_l2=MODELS[method](**MODEL_PARAMETERS[method]))
            try:
                icll.fit(X_train, y_train)
                icll_prob = icll.predict_proba(X_test)
                icll_l2_prob = icll.predict_proba_l2(X_test)
                icll_l1_prob = icll.predict_proba_l1(X_test)
            except NoGreyZoneError:
                no_grey_zone[file] = AUC(y_true=y_test, y_score=baseline_prob)
                print(f'NoGreyZoneError error. Falling to baseline')
                icll_prob = np.repeat(np.nan, len(baseline_prob))
                icll_l2_prob = np.repeat(np.nan, len(baseline_prob))
                icll_l1_prob = np.repeat(np.nan, len(baseline_prob))
                pass

            print('Training tsl r')
            icllr_tt = ICLL(apply_resample_l1=True,
                            apply_resample_l2=True,
                            model_l1=MODELS[method](**MODEL_PARAMETERS[method]),
                            model_l2=MODELS[method](**MODEL_PARAMETERS[method]))
            try:
                icllr_tt.fit(X_train, y_train)
                icllr_tt_prob = icllr_tt.predict_proba(X_test)
            except NoGreyZoneError:
                icllr_tt_prob = np.repeat(np.nan, len(baseline_prob))

            print('Training tsl r')
            icllr_tf = ICLL(apply_resample_l1=True,
                            apply_resample_l2=False,
                            model_l1=MODELS[method](**MODEL_PARAMETERS[method]),
                            model_l2=MODELS[method](**MODEL_PARAMETERS[method]))
            try:
                icllr_tf.fit(X_train, y_train)
                icllr_tf_prob = icllr_tf.predict_proba(X_test)
            except NoGreyZoneError:
                icllr_tf_prob = np.repeat(np.nan, len(baseline_prob))

            print('Training tsl r')
            icllr_ft = ICLL(apply_resample_l1=False,
                            apply_resample_l2=True,
                            model_l1=MODELS[method](**MODEL_PARAMETERS[method]),
                            model_l2=MODELS[method](**MODEL_PARAMETERS[method]))
            try:
                icllr_ft.fit(X_train, y_train)
                icllr_ft_prob = icllr_ft.predict_proba(X_test)
            except NoGreyZoneError:
                icllr_ft_prob = np.repeat(np.nan, len(baseline_prob))

            baseline_sc = AUC(y_true=y_test, y_score=baseline_prob)
            baseline_svm_sc = AUC(y_true=y_test, y_score=baseline_svm_prob)
            baseline_lr_sc = AUC(y_true=y_test, y_score=baseline_lr_prob)
            clf_smote_sc = AUC(y_true=y_test, y_score=clf_smote_prob)
            clf_nearmiss_sc = AUC(y_true=y_test, y_score=clf_nearmiss_prob)
            clf_ro_sc = AUC(y_true=y_test, y_score=clf_ro_prob)
            clf_ru_sc = AUC(y_true=y_test, y_score=clf_ru_prob)
            clf_oss_sc = AUC(y_true=y_test, y_score=clf_oss_prob)
            clf_adasyn_sc = AUC(y_true=y_test, y_score=clf_ada_prob)
            cure_sc = AUC(y_true=y_test, y_score=cure_prob)
            seqll_sc = AUC(y_true=y_test, y_score=icll_prob)
            seqll_l2_sc = AUC(y_true=y_test, y_score=icll_l2_prob)
            seqll_l1_sc = AUC(y_true=y_test, y_score=icll_l1_prob)
            seqllr_tt_sc = AUC(y_true=y_test, y_score=icllr_tt_prob)
            seqllr_tf_sc = AUC(y_true=y_test, y_score=icllr_tf_prob)
            seqllr_ft_sc = AUC(y_true=y_test, y_score=icllr_ft_prob)
            brf_sc = AUC(y_true=y_test, y_score=balancedrf_prob)

            scores.append(
                {
                    'NoResample': baseline_sc,
                    'NoResampleSVM': baseline_svm_sc,
                    'NoResampleLR': baseline_lr_sc,
                    'SMOTE': clf_smote_sc,
                    'NearMiss': clf_nearmiss_sc,
                    'OSS': clf_oss_sc,
                    'ADASYN': clf_adasyn_sc,
                    'RO': clf_ro_sc,
                    'RU': clf_ru_sc,
                    'CURE': cure_sc,
                    'ICLL': seqll_sc,
                    'ICLL+SMOTE': seqllr_tt_sc,
                    'ICLL+SMOTE(L1)': seqllr_tf_sc,
                    'ICLL+SMOTE(L2)': seqllr_ft_sc,
                    'ICLL(L2)': seqll_l2_sc,
                    'ICLL(L1)': seqll_l1_sc,
                    'BalancedRF': brf_sc,
                }
            )

        avg_scr = pd.DataFrame(scores).mean()
        scores_across_datasets.append(avg_scr)

        scr = pd.DataFrame(scores_across_datasets)
        print(scr.rank(axis=1, ascending=False).mean())

        scr_hard = scr.loc[scr['NoResample'] < 0.95, :]
        scr.to_csv('results.csv', index=False)
        scr_hard.to_csv('results_hard.csv', index=False)


if __name__ == '__main__':
    main()
