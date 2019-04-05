import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

class RF():
    def __init__(self,model_name):

        self.trained_model_path='trained_model/'+model_name+'.sav'

    def train(self,tr_data,tr_label):
        model =RandomForestClassifier(n_estimators=50, min_samples_split=4, \
                                      class_weight='balanced',random_state=1,n_jobs=5)
        model.fit(tr_data,tr_label)


        predictions=model.predict(tr_data)

        result=model.score(tr_data,tr_label)
        print 'Train Accuracy: {0:02f}'.format(result)

        pickle.dump(model, open(self.trained_model_path,'wb'))

    def test(self,te_data,te_label):

        model=pickle.load(open(self.trained_model_path,'rb'))
        result=model.score(te_data,te_label)

        print 'Test Accuracy: {0:02f}'.format(result)



