from  Classifier.Data_processing import make_train_test
import pandas as pd
import copy
import pickle
from Explanation.LIME.LIME_tabular import *
import itertools
import xgboost as xgb
from pandas import DataFrame as df
from tqdm import tqdm
class lime_libraray():
    def __init__(self,in_path,model):
        self.in_path=in_path
        self.model=model

    def build(self):
        tr_data, te_data, tr_label, te_label = make_train_test(self.in_path)

        # extract feature names and class names from heloc dataset
        feature_names = list()
        load_df = pd.read_csv(self.in_path)
        class_name = list()


        for col in load_df.columns:
            feature_names.append(col)


        if 'Random_Forest' in self.model:
            model_n = 'RF'
        elif 'XGboost' in self.model:
            model_n = 'XGboost'



        # delete label feature name
        if 'HELOC' in self.in_path:
            del feature_names[0]
            class_name.append('Bad')
            class_name.append('Good')
            df_explain_valueSim = pd.read_csv("Explain_te_data/heloc/" + model_n + '_explain_valueSim.csv')
            df_explain_probSim = pd.read_csv("Explain_te_data/heloc/" + model_n + '_explain_probSim.csv')
            df_explain_random = pd.read_csv("Explain_te_data/heloc/" + model_n + '_explain_random.csv')


        elif 'UCI_Credit_Card' in self.in_path:
            del feature_names[23]
            class_name.append('payment')
            class_name.append('not payment')
            df_explain_valueSim = pd.read_csv("Explain_te_data/UCI_Credit_Card/" + model_n + '_explain_valueSim.csv')
            df_explain_probSim = pd.read_csv("Explain_te_data/UCI_Credit_Card/" + model_n + '_explain_probSim.csv')
            df_explain_random = pd.read_csv("Explain_te_data/UCI_Credit_Card/" + model_n + '_explain_random.csv')


        explainer = lime_tabular(tr_data, feature_names=feature_names, \
                                 class_names=class_name, discretize_continuous=False)

        model = pickle.load(open(self.model, 'rb'))




        exp_list = [df_explain_valueSim,df_explain_probSim,df_explain_random]

        exp_n_list = ['valueSim','probSim','random']

        for i in tqdm(range(3)):
            for row in exp_list[i].iterrows():
                index, value = row

                tmp = np.asarray(value, dtype=np.float32)
                feature_size = len(feature_names)
                tmp = tmp.reshape(1, feature_size)

                if 'XGboost' in self.model:
                    predict_fn_xgb = lambda x: model.predict_proba(df(data=x, columns=feature_names, \
                                                                      index=[np.arange(x.shape[0])])).astype(float)
                    exp = explainer.explain_instance(tmp, predict_fn_xgb, num_features=feature_size, top_labels=1)

                elif 'Random_Forest' in self.model:
                    exp = explainer.explain_instance(tmp, model.predict_proba, num_features=feature_size, top_labels=1)

                else:
                    raise ValueError('exp must be "XGboost or Random_Forest')

                if 'HELOC' in self.in_path:

                    sav_name = 'result/heloc/LIME_result_'+model_n+'_' + str(index) + '(' + exp_n_list[i] + ')' + '.html'
                elif 'UCI_Credit_Card' in self.in_path:
                    sav_name = 'result/UCI_Credit_Card/LIME_result_'+model_n+'_' + str(index) + '(' + exp_n_list[i] + ')' + '.html'

                exp.save_to_file(sav_name)

        print ('\nComplete!')
