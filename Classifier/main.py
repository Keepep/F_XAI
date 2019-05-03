from RF import RF
from XGboost import XGB

from Data_processing import make_train_test
from Data_processing import fico_data_preprocessing
from Data_processing import UCI_data_preprocessing
from Data_processing import make_explain_data
import os

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier for financial data')
    parser.add_argument('--data_path', type=str, default='../Dataset/heloc_dataset_v1.csv', help='Dataset path')
    parser.add_argument('--model_name', type=str, default='Random_Forest', help='Model Name')

    args = parser.parse_args()

    #preprocessing
    if 'heloc' in args.data_path:
        file_path=fico_data_preprocessing(args.data_path)
    elif 'UCI_Credit_Card' in args.data_path:
        file_path=UCI_data_preprocessing(args.data_path)
    elif 'statlog' in args.data_path:
        file_path = args.data_path

    #make_test_data(file_path)============================================= Modify data path.
    tr_data,te_data, tr_label,te_label=make_train_test(file_path)


    if args.model_name =="XGboost":
        model=XGB(args.model_name)
    elif args.model_name == "Random_Forest":
        model=RF(args.model_name)


    model.train(tr_data,tr_label)
    model.test(te_data,te_label)

    #generating data for explaining using test dataset
    prob=model.get_prob(te_data)
    make_explain_data(te_data,prob,args.model_name,args.data_path)


