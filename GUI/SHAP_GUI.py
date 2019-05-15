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
import copy
class shap_library():
    def __init__(self,dataset, in_path, model):
        self.dataset=dataset
        self.in_path = in_path
        self.model = model


    def build_for_FICO(self):
        tr_data, te_data, tr_label, te_label = make_train_test(self.dataset)


        f, l = get_col_name(self.in_path, tr_data)


        # the index of instances
        ins = 2
        explainer = KernelExplainer(self.model.predict_proba, tr_data, nsamples=100, link="identity", keep_index=True)

        # Get shapley values
        load_df=pd.read_csv(self.in_path)


        shap_value = explainer.shap_values(load_df[0:1], nsamples=200)
        print shap_value
        if(explainer.expected_value[0] > explainer.expected_value[1]):
            target = 0
        else:
            target = 1
        neg, pos, output =force_plot(explainer.expected_value[target], shap_value[target], load_df.iloc[0], matplotlib=True, link="identity")

        print('Expect: {0}, Probability: {1}'.format(target, explainer.expected_value[target]))
        print('[0] Contribution [1] Real value [2] Name of feature')
        print('Negative')
        print(neg)
        print('Positive')
        print(pos)




        compr = np.empty(pos.shape[0])
        pos_copy=copy.copy(pos)
        size=pos.shape[0]
        for j in range(size):
            if j==0:
                cont_pos=output-float(pos[j][0])
            else:
                cont_pos=float(pos[j-1][0])-float(pos[j][0])

            compr[j]=cont_pos
            pos_copy[j][0] = cont_pos



        for j in range(size):
            argmax=np.argmax(compr)

            pos[j][0]= pos_copy[argmax][0]
            pos[j][1] = pos_copy[argmax][1]
            pos[j][2] = pos_copy[argmax][2]
            compr[argmax]=0

        print pos
        print output

        compr = np.empty(neg.shape[0])

        neg_copy=copy.copy(neg)
        size = neg.shape[0]
        for j in range(size):
            if j==0:
                cont_neg=float(neg[j][0])-output
            else:
                cont_neg=float(neg[j][0])-float(neg[j-1][0])
            compr[j]=cont_neg

            neg_copy[j][0] = cont_neg

        print neg_copy
        for j in range(size):
            argmax=np.argmax(compr)

            neg[j][0]=neg_copy[argmax][0]
            neg[j][1] = neg_copy[argmax][1]
            neg[j][2] = neg_copy[argmax][2]
            compr[argmax]=0

        print neg
        return pos ,neg
