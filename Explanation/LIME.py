import lime.lime_tabular
from utils import *
import lime
from  Classifier.Data_processing import make_train_test
import pandas as pd
import copy
import numpy as np
import pickle
import sklearn
class lime_libraray():
    def __init__(self,in_path,model):
        self.in_path=in_path
        self.model=model

    def tabular(self):
        tr_data, te_data, tr_label, te_label = make_train_test(self.in_path)

        load_df = pd.read_csv(self.in_path)
        test=list()
        for col in load_df.columns:
            test.append(col)

        feature_names=copy.copy(test)

        del feature_names[0]
        class_name=list()
        class_name.append('Good')
        class_name.append('Bad')

        #print np.shape(tr_data)
        #print feature_names

        explainer = lime.lime_tabular.LimeTabularExplainer(tr_data, feature_names=feature_names,\
                                                class_names=class_name, discretize_continuous=False)

        i = np.random.randint(0, te_data.shape[0])
        model = pickle.load(open(self.model, 'rb'))
        result = model.score(te_data, te_label)
        tmp=list()
        for row in te_data.iterrows():
            index,value=row
            tmp.append(value.tolist())
            break
        tmp= np.asarray(tmp, dtype=np.float32)
        tmp=tmp.flatten()
        #print model.predict_proba(tmp)
        #predict_fn = lambda x: model.predict_proba(encoder.transform(x))
        exp = explainer.explain_instance(tmp,model.predict_proba , num_features=23, top_labels=1)
        exp.show_in_notebook()
