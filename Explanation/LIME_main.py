from  Classifier.Data_processing import make_train_test
import pandas as pd
import copy
import numpy as np
import pickle
from Explanation.LIME.LIME_tabular import *

class lime_libraray():
    def __init__(self,in_path,model):
        self.in_path=in_path
        self.model=model

    def build_for_FICO(self):
        tr_data, te_data, tr_label, te_label = make_train_test(self.in_path)

        load_df = pd.read_csv(self.in_path)
        feature_names=list()

        for col in load_df.columns:
            feature_names.append(col)

        #feature_names=copy.copy(test)

        del feature_names[0]

        class_name=list()
        class_name.append('Good')
        class_name.append('Bad')


        explainer =lime_tabular(tr_data, feature_names=feature_names,\
                                                class_names=class_name, discretize_continuous=False)

        i = np.random.randint(0, te_data.shape[0])
        model = pickle.load(open(self.model, 'rb'))
        result = model.score(te_data, te_label)
        tmp=list()
        i=1

        for row in te_data.iterrows():
            sav_name = 'LIME_result/result_' + str(i) + '.html'
            index,value=row

            tmp.append(value.tolist())
            tmp = np.asarray(tmp, dtype=np.float32)
            tmp = tmp.flatten()
            exp = explainer.explain_instance(tmp, model.predict_proba, num_features=23, top_labels=1)
            exp.save_to_file(sav_name)
            tmp=list()
            i=i+1

            if i==10:
                break