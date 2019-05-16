import xgboost as xgb
import pickle
import os
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
class XGB():
    def __init__(self, model_name, file_path):

        data_name = 'XGboost'

        # get filename and remove .csv
        self.file_name = os.path.basename(file_path)[:-4]
        self.trained_model_path = 'Classifier/trained_model/' + data_name + '_' + self.file_name + '.sav'

    def train(self, tr_data, tr_label):
        model = xgb.XGBClassifier(objective="binary:logistic", random_state=112)
        model.fit(tr_data, tr_label)

        result = model.score(tr_data, tr_label)

        # print 'Train Accuracy: {0:02f}'.format(result)
        pickle.dump(model, open(self.trained_model_path,'wb'))

        self.train_fpr, self.train_tpr, roc = self.get_roc(tr_data, tr_label)
        print 'Train ROC: '+str(roc)
        print 'Train AR: '+str(roc*2-1)

    def test(self, te_data, te_label):

        model=pickle.load(open(self.trained_model_path,'rb'))
        result=model.score(te_data,te_label)
        self.test_fpr, self.test_tpr, roc = self.get_roc(te_data, te_label)
        print 'Test ROC: ' + str(roc)
        print 'Test AR: ' + str(roc * 2 - 1)
        # print 'Test Accuracy: {0:02f}'.format(result)

    def get_prob(self, te_data):
        model=pickle.load(open(self.trained_model_path,'rb'))
        prob=model.predict_proba(te_data)
        return prob

    def get_roc(self, x, y):
        if 'HELOC' in self.file_name or 'heloc' in self.file_name:
            y = y.replace(['Good', 'Bad'], [1, 0])
        fpr, tpr, thresholds = roc_curve(y, self.get_prob(x)[:, 1])
        score = auc(fpr, tpr)
        return fpr, tpr, score

    def draw_roc_curve(self):
        fig = plt.figure()
        plt.plot([0, 1], [0, 1], label='0.5')
        plt.plot(self.train_fpr, self.train_tpr, label='train')
        plt.plot(self.test_fpr, self.test_tpr, label='test')
        plt.legend(loc='best')
        plt.title('AUROC (Area Under the Curve)')
        plt.xlabel('FPR (False Positive Ratio)')
        plt.ylabel('TPR (True Positive Ratio)')

        return fig