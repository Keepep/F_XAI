import pickle
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
class RF():
    def __init__(self, model_name, file_path):
        if model_name == 'XGboost':
            data_name = 'XGboost'
        elif model_name == 'Random_Forest':
            data_name = 'RF'
        # get filename and remove .csv
        file_name = os.path.basename(file_path)[:-4]
        self.trained_model_path = 'Classifier/trained_model/' + data_name + '_' + file_name + '.sav'


    def train(self, tr_data, tr_label):
        model =RandomForestClassifier(n_estimators=50, min_samples_split=4, \
                                      class_weight='balanced',random_state=1,n_jobs=5)
        model.fit(tr_data, tr_label)

        result = model.score(tr_data, tr_label)
        # print 'Train Accuracy: {0:02f}'.format(result)

        pickle.dump(model, open(self.trained_model_path,'wb'))

    def test(self, te_data, te_label):

        model = pickle.load(open(self.trained_model_path,'rb'))
        result = model.score(te_data, te_label)

        # print 'Test Accuracy: {0:02f}'.format(result)

    def get_prob(self, te_data):
        model=pickle.load(open(self.trained_model_path,'rb'))
        prob=model.predict_proba(te_data)
        return prob

