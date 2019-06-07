from sklearn import linear_model
import matplotlib.pyplot as plt
import numpy as np
import sys
class LogisticRegression():
    def __init__(self, data_path):
        self.model = None
        self.data_path = data_path
        self.features = None
        self.label = None
    def train(self, x, y):
        self.model = linear_model.LogisticRegression(solver='lbfgs', max_iter=15000)
        self.model.fit(x, y)
        self.features, self.label = self.get_col_name(self.data_path, x)
    def eval(self, x, y):
        if(self.model):
            pred = self.model.predict(x)
            # print('Prediction of interpretable model = {0}'.format(pred))

            score = self.model.score(x, y)
            print('Accuracy of interpretable model = {0}'.format(score))
        else:
            print('Training Classification model first')
    def importance(self, summary_plot=None, instance_plot=None, ins=None):
        f_i = np.reshape(np.argsort(-self.model.coef_), [-1])
        feature_im = np.reshape([self.model.coef_[0, i] for i in f_i], [-1])
        features = [self.features[i] for i in f_i]
        if summary_plot:
            x = np.arange(len(feature_im))
            fig1 = plt.figure()
            plt.bar(x, feature_im)
            plt.xticks(x, features, rotation=90)
            plt.title('Overall feature importance of global surrogate')
            plt.show()

        if instance_plot:
            if ins is None:
                print('Instance data is needed for plotting')
                return 0
            ins_im = np.reshape(self.model.coef_, [-1]) * ins
            i = np.argsort(ins_im)
            features = [self.features[ix] for ix in i]
            x = [ins_im[ix] for ix in i]
            print('coef {0}'.format(np.reshape(self.model.coef_, [-1])))
            print('ins {0}'.format(ins))
            print('ins_im {0}'.format(ins_im))
            fig2 = plt.figure()
            plt.barh(np.arange(len(ins_im)), x)
            plt.yticks(np.arange(len(ins_im)), features)
            plt.title('Instance feature importance of global surrogate')
            plt.show()

        return feature_im

    def get_col_name(self, data_path, x):

        if 'heloc' in data_path or 'HELOC' in data_path:
            label_name = 'RiskPerformance'
        elif 'UCI_Credit' in data_path:
            label_name = 'default.payment.next.month'
        elif 'statlog' in data_path:
            label_name = 'Credit'
        elif 'LoanStat' in data_path:
            label_name = 'loan_status'
        else:
            print('label name not vaild!')
            sys.exit(0)
        features = [c for c in x.columns if c != label_name]
        return features, label_name
