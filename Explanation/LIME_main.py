from  Classifier.Data_processing import make_train_test
import pandas as pd
import pickle
from Explanation.LIME.LIME_tabular import *
import xgboost as xgb
from pandas import DataFrame as df
from tqdm import tqdm
<<<<<<< HEAD
import sys, os
import torch
from torch.autograd import Variable



def cuda_available():
    use_cuda = torch.cuda.is_available()
    return use_cuda
=======
import fnmatch
import sys, os
>>>>>>> origin/master

class lime_libraray():
    def __init__(self, in_path, model_name):
        self.in_path=in_path
        self.model_name = model_name
        if 'heloc' in self.in_path or 'HELOC' in self.in_path:
            self.data_name = 'heloc'
            self.feature_i = 0
            self.class_label=['Bad', 'Good']
        elif 'UCI_Credit_Card' in self.in_path:
            self.data_name = 'UCI_Credit_Card'
            self.feature_i = 23
            self.class_label = ['payment', 'not payment']
        elif 'statlog' in self.in_path:
            self.data_name = 'statlog'
            self.feature_i = 20
            self.class_label = ['Bad', 'Good']
        else:
            # print 'There {0} data is not valid path'.format(self.in_path)
            sys.exit(0)

    def build(self):
        tr_data, te_data, tr_label, te_label = make_train_test(self.in_path)

        # extract feature names and class names from heloc dataset
        feature_names = list()
        load_df = pd.read_csv(self.in_path)
        class_name = list()

        for col in load_df.columns:
            feature_names.append(col)

        if 'Random_Forest' in self.model_name:
            model_n = 'RF'
<<<<<<< HEAD
            ext='.sav'
        elif 'XGboost' in self.model_name:
            model_n = 'XGboost'
            ext='.sav'
        elif 'GBM' in self.model_name:
            model_n= 'GBM'
            ext = '.sav'
        elif 'MLP' in self.model_name:
            model_n='MLP'
            ext='.pt'
        del feature_names[self.feature_i]
        class_name.append(self.class_label[0])
        class_name.append(self.class_label[1])

=======
        elif 'XGboost' in self.model_name:
            model_n = 'XGboost'

        del feature_names[self.feature_i]
        class_name.append(self.class_label[0])
        class_name.append(self.class_label[1])
>>>>>>> origin/master
        df_explain_valueSim = pd.read_csv("Explain_te_data/"+ self.data_name +"/" + model_n + '_explain_valueSim.csv')
        df_explain_probSim = pd.read_csv("Explain_te_data/"+ self.data_name +"/" + model_n + '_explain_probSim.csv')
        df_explain_random = pd.read_csv("Explain_te_data/"+ self.data_name +"/" + model_n + '_explain_random.csv')

        explainer = lime_tabular(tr_data, feature_names=feature_names, \
                                 class_names=class_name, discretize_continuous=False)
        # get file name and remove .csv
        file_name = os.path.basename(self.in_path)[:-4]
<<<<<<< HEAD
        saved_model_path = 'Classifier/trained_model/' + model_n + '_' + file_name +ext

        model = pickle.load(open(saved_model_path, 'rb'))
        if ext=='.pt':
            model = torch.load(saved_model_path)
            model.eval()

            if cuda_available():
                model.cuda()


=======
        saved_model_path = 'Classifier/trained_model/' + model_n + '_' + file_name +'.sav'
        model = pickle.load(open(saved_model_path, 'rb'))
>>>>>>> origin/master

        exp_list = [df_explain_valueSim,df_explain_probSim,df_explain_random]

        exp_n_list = ['valueSim','probSim','random']

        for i in tqdm(range(3)):
            for row in exp_list[i].iterrows():
                index, value = row

                tmp = np.asarray(value, dtype=np.float32)
                feature_size = len(feature_names)
                tmp = tmp.reshape(1, feature_size)

                if 'XGboost' in self.model_name:
                    predict_fn_xgb = lambda x: model.predict_proba(df(data=x, columns=feature_names, \
                                                                      index=[np.arange(x.shape[0])])).astype(float)
                    exp = explainer.explain_instance(tmp, predict_fn_xgb, num_features=feature_size, top_labels=1)

                elif 'Random_Forest' in self.model_name:
                    exp = explainer.explain_instance(tmp, model.predict_proba, num_features=feature_size, top_labels=1)
                elif 'GBM' in self.model_name:
                    exp = explainer.explain_instance(tmp, model.predict_proba, num_features=feature_size, top_labels=1)
                elif 'MLP' in self.model_name:
                    predict_fn_mlp=lambda x: torch.nn.Softmax()(model.forward(Variable(torch.from_numpy(x).float().cuda()) if cuda_available() else \
                        Variable(torch.from_numpy(x).float()))).cpu().data.numpy()
                    exp = explainer.explain_instance(tmp, predict_fn_mlp, num_features=feature_size, top_labels=1)
                else:
                    raise ValueError('exp must be "XGboost or Random_Forest')

                sav_name = 'result/'+ self.data_name +'/LIME_result_'+model_n+'_' + str(index) + '(' + exp_n_list[i] + ')' + '.html'
                exp.save_to_file(sav_name)

        print ('\nComplete!')
