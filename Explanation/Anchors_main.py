from  Classifier.Data_processing import make_train_test
import pandas as pd
import copy
import numpy as np
import pickle
#from __future__ import print_function
import numpy as np
np.random.seed(1)
import sys
import sklearn
import sklearn.ensemble
from sklearn import linear_model
#load_ext autoreload
#autoreload 2
import Explanation.Anchors.utils as utils
import Explanation.Anchors.anchor_tabular as anchor_tabular
import pickle
import xgboost as xgb
import Classifier


class anchors_libraray():

    def __init__(self,in_path,model):
        self.in_path=in_path
        self.model=model

    def anchor(self):
        # make sure you have adult/adult.data inside dataset_folder
        dataset_folder = '/home/kidon/Downloads/anchor-master/anchor/datasets'
        dataset = utils.load_dataset('FICO', balance=True, dataset_folder=dataset_folder)

        explainer = anchor_tabular.AnchorTabularExplainer(dataset.class_names, dataset.feature_names, dataset.data,
                                                          dataset.categorical_names)
        explainer.fit(dataset.train, dataset.labels_train, dataset.validation, dataset.labels_validation)

        if self.model == "XGboost":
            c = xgb.XGBClassifier(objective="binary:logistic")
            c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)
        elif self.model == "Random_Forest":
            c = sklearn.ensemble.RandomForestClassifier(n_estimators=50, n_jobs=5)
            c.fit(explainer.encoder.transform(dataset.train), dataset.labels_train)

        # evaluation
        predict_fn = lambda x: c.predict(explainer.encoder.transform(x))
        print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, predict_fn(dataset.train)))
        print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, predict_fn(dataset.test)))

        idx = 0
        np.random.seed(1)
        print('Prediction: ', explainer.class_names[predict_fn(dataset.test[idx].reshape(1, -1))[0]])
        exp = explainer.explain_instance(dataset.test[idx], c.predict, threshold=0.95)

        print('Anchor: %s' % (' AND '.join(exp.names())))
        print('Precision: %.2f' % exp.precision())
        print('Coverage: %.2f' % exp.coverage())

        # Get test examples where the anchora pplies
        fit_anchor = np.where(np.all(dataset.test[:, exp.features()] == dataset.test[idx][exp.features()], axis=1))[0]
        print('Anchor test coverage: %.2f' % (fit_anchor.shape[0] / float(dataset.test.shape[0])))
        print('Anchor test precision: %.2f' % (
            np.mean(predict_fn(dataset.test[fit_anchor]) == predict_fn(dataset.test[idx].reshape(1, -1)))))

        print('Partial anchor: %s' % (' AND '.join(exp.names(1))))
        print('Partial precision: %.2f' % exp.precision(1))
        print('Partial coverage: %.2f' % exp.coverage(1))

        fit_partial = np.where(np.all(dataset.test[:, exp.features(1)] == dataset.test[idx][exp.features(1)], axis=1))[
            0]
        print('Partial anchor test precision: %.2f' % (
            np.mean(predict_fn(dataset.test[fit_partial]) == predict_fn(dataset.test[idx].reshape(1, -1)))))
        print('Partial anchor test coverage: %.2f' % (fit_partial.shape[0] / float(dataset.test.shape[0])))
