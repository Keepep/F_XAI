from  Classifier.Data_processing import make_train_test
import pandas as pd
import copy
import numpy as np
import pickle
from Explanation.LIME.LIME_tabular import *

class lime_libraray():
    def __init__(self,dataset,in_path,model):
        self.dataset=dataset
        self.in_path=in_path
        self.model=model

    def build_for_FICO(self):
        tr_data, te_data, tr_label, te_label = make_train_test(self.dataset)
        load_df = pd.read_csv(self.in_path)
        feature_names=list()

        for col in load_df.columns:
            feature_names.append(col)

        #feature_names=copy.copy(test)


        class_name=list()
        class_name.append('Good')
        class_name.append('Bad')


        explainer =lime_tabular(tr_data, feature_names=feature_names,\
                                                class_names=class_name, discretize_continuous=False)

        model = self.model
        tmp=load_df.values

        exp = explainer.explain_instance(tmp, model.predict_proba, num_features=22, top_labels=1)
        """
        if 1 in exp.local_exp.keys():
            for j in range(10):
                print('Prediction:Bad' ,'Feature_name:', feature_names[exp.local_exp.values()[1][j][0]], 'Attribute:', exp.local_exp.values()[1][j][1])

        if 0 in exp.local_exp.keys():

            for j in range(10):
                print('Prediction:Good' ,'Feature_name:', feature_names[exp.local_exp.values()[0][j][0]], 'Attribute:', exp.local_exp.values()[0][j][1])
        """

        return feature_names, exp
