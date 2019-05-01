import xgboost as xgb
import pickle


import numpy as np
class XGB():
    def __init__(self,model_name):

        self.trained_model_path='trained_model/'+model_name+'.sav'

    def train(self,tr_data,tr_label):
        model =xgb.XGBClassifier(objective="binary:logistic", random_state=112)
        model.fit(tr_data,tr_label)

        result=model.score(tr_data,tr_label)
        print 'Train Accuracy: {0:02f}'.format(result)

        pickle.dump(model, open(self.trained_model_path,'wb'))

    def test(self,te_data,te_label):

        model=pickle.load(open(self.trained_model_path,'rb'))
        result=model.score(te_data,te_label)

        print 'Test Accuracy: {0:02f}'.format(result)



    def get_prob(self,te_data):
        model=pickle.load(open(self.trained_model_path,'rb'))
        prob=model.predict_proba(te_data)

        return prob
