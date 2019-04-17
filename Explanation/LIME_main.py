from  Classifier.Data_processing import make_train_test
import pandas as pd
import copy
from numpy import loadtxt
import pickle
from Explanation.LIME.LIME_tabular import *
import itertools
import xgboost as xgb
from sklearn.model_selection import train_test_split
from pandas import DataFrame as df

class lime_libraray():
    def __init__(self,in_path,model):
        self.in_path=in_path
        self.model=model

    def build_for_FICO(self):
        tr_data, te_data, tr_label, te_label = make_train_test(self.in_path)

        tr_data_good=copy.copy(tr_data)


        for index,value in itertools.izip(tr_label.keys(),tr_label):

            if value == 'Good':
                tr_data_good.drop(index,0)



        load_df = pd.read_csv(self.in_path)

        feature_names=list()

        for col in load_df.columns:
            feature_names.append(col)

        del feature_names[0]

        class_name=list()
        class_name.append('Bad')
        class_name.append('Good')


        explainer =lime_tabular(tr_data, feature_names=feature_names,\
                                                class_names=class_name, discretize_continuous=False)

        model = pickle.load(open(self.model, 'rb'))
        i=1

        predict_fn_xgb= lambda x: model.predict_proba(df(data=x,columns=feature_names,index=[np.arange(x.shape[0])]\
                                                         )).astype(float)

        for row in tr_data.iterrows():
            sav_name = 'result/LIME_result/result_' + str(i) + '.html'

            index,value=row
            tmp = np.asarray(value, dtype=np.float32)
            tmp = tmp.reshape(1,23)


            if 'XGboost' in self.model:
                exp = explainer.explain_instance(tmp, predict_fn_xgb, num_features=23,top_labels=1)
            elif 'Random_Forest' in self.model :
                exp = explainer.explain_instance(tmp, model.predict_proba, num_features=23,top_labels=1)
            else:
                raise ValueError('exp must be "XGboost or Random_Forest')
            exp.save_to_file(sav_name)

            i=i+1

            if i==10:
                break
        print ('Complete!')