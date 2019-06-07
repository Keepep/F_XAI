import argparse
from Explanation.LIME_main import *
from Explanation.Anchors_main import *
from Explanation.SHAP_main import *
from Classifier.Data_processing import make_train_test
from Classifier.Data_processing import fico_data_preprocessing
from Classifier.Data_processing import UCI_data_preprocessing
from Classifier.Data_processing import make_explain_data
from Classifier.RF import RF
from Classifier.XGboost import XGB
from Classifier.GradientBoosting import GBM
from Classifier.MLP import MLP

from Explanation.global_surrogate_main import LogisticRegression

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XAI for financial data')
    parser.add_argument('--data_path', type=str, default='Dataset/UCI_Credit_Card.csv', help='Dataset path')
    parser.add_argument('--model_name', type=str, default='XGboost', help='Model name')
    parser.add_argument('--Explanation',type=str, default='Anchors', help='Explanation method')
    args = parser.parse_args()

    # preprocessing
    if 'heloc' in args.data_path:
        file_path = fico_data_preprocessing(args.data_path)
    elif 'UCI_Credit_Card' in args.data_path:
        file_path = UCI_data_preprocessing(args.data_path)
    elif 'statlog' in args.data_path:           # Its Prob[0:,1] is more than prob[:,0]
        file_path = args.data_path              # Procedure getting Probsim stuck in problem (<= 0.5)
    elif 'LoanStats' in args.data_path:
        file_path = args.data_path

    # make_test_data(file_path)
    tr_data, te_data, tr_label, te_label = make_train_test(file_path, True)

    if "XGboost" == args.model_name:
        model = XGB(args.model_name, file_path)
    elif "Random_Forest" == args.model_name:
        model = RF(args.model_name, file_path)
    elif "GBM" == args.model_name:
        model=GBM(args.model_name, file_path )
    elif "MLP" == args.model_name:
        model = MLP(args.model_name, file_path)


    model.train(tr_data, tr_label)
    model.test(te_data, te_label)
    f_roc = model.draw_roc_curve()
    f_roc.show()

    # generating data for explaining using test dataset
    prob = model.get_prob(te_data)
    pred = model.get_pred(tr_data)
    make_explain_data(te_data, prob, args.model_name, file_path)


    if args.Explanation == 'lime':
        lime=lime_libraray(file_path, args.model_name)
        #lime.build_for_UCI_Credit_Card()
        lime.build()

    elif args.Explanation == 'Anchors':
        anchor =anchors_libraray(file_path, args.model_name)
        anchor.anchor()

    elif args.Explanation == 'Shap':
        shap = shap_library(file_path, args.model_name)
        shap.build_for_FICO()

    # Global surrogate
    global_model = LogisticRegression(args.data_path)

    global_model.train(tr_data, pred)
    global_model.eval(te_data, te_label)

    # How to show feature importance of Logistic Regresssion
    importance = global_model.importance(summary_plot=True, instance_plot=True, ins=np.array(te_data.iloc[0]))