from RF import RF
from XGboost import XGB
from Data_processing import make_train_test
import argparse
from test import test
import Data_processing
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier for financial data')
    parser.add_argument('--data_path', type=str, default='Dataset/heloc_dataset_v1.csv', help='Dataset path')
    parser.add_argument('--model_name', type=str, default='XGboost', help='Model Name')

    args = parser.parse_args()


    tr_data,te_data, tr_label,te_label=make_train_test(args.data_path)

    if args.model_name =="XGboost":
        model=XGB(args.model_name)
    elif args.model_name == "Random_Forest":
        model=RF(args.model_name)
    elif args.model_name == "anchorsclassfier_model":
        model=test(args.model_name)
    model.train(tr_data,tr_label)
    model.test(te_data,te_label)
