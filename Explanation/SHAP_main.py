from Explanation.SHAP.explainers.kernel import KernelExplainer
from Explanation.SHAP.plots.summary import  summary_plot
from Explanation.SHAP.plots.force import *
import pickle
from Classifier.Data_processing import make_train_test
from  Classifier.Data_processing import get_col_name
import os
import xgboost as xgb
import pandas as pd
import matplotlib as plt
class shap_library():
    def __init__(self, in_path, model):
        self.in_path = in_path
        self.model = model
        if 'Random_Forest' == self.model:
            model_name = 'RF'
        elif 'XGboost' == self.model:
            model_name = 'XGboost'
        self.model_path = 'Classifier/trained_model/'+model_name+'_'+os.path.basename(self.in_path)[:-4]+'.sav'

    def build_for_FICO(self):
        tr_data, te_data, tr_label, te_label = make_train_test(self.in_path)
        model = pickle.load(open(self.model_path, 'rb'))

        f, l = get_col_name(self.in_path, tr_data)
        # the index of instances
        ins = 2
        explainer = KernelExplainer(model.predict_proba, tr_data, nsamples=100, link="logit", keep_index=True)

        # Get shapley values
        shap_value = explainer.shap_values(te_data[ins:ins+1], nsamples=100)
        print(shap_value)
        print(sum(shap_value[0][0]))
        print(sum(shap_value[1][0]))

        if(explainer.expected_value[0] > explainer.expected_value[1]):
            target = 0
        else:
            target = 1
        neg, pos, out =force_plot(explainer.expected_value[target], shap_value[target], te_data.iloc[ins], matplotlib=True, link="logit")
        feature_importance = model.feature
        print('Expect: {0}, Probability: {1}'.format(target, explainer.expected_value[target]))
        print('[0] Contribution [1] Real value [2] Name of feature')
        print('Negative')
        print(neg)
        print('Positive')
        print(pos)
        print(out)
