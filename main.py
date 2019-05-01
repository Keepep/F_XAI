
import argparse
from Explanation.LIME_main import *
from Explanation.Anchors_main import *
from Explanation.SHAP_main import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='XAI for financial data')
    parser.add_argument('--data_path', type=str, default='Dataset/HELOC_allRemoved.csv', help='Dataset path')
    parser.add_argument('--model_name', type=str, default='Classifier/trained_model/Random_Forest.sav', help='Model Name')
    parser.add_argument('--Explanation',type=str, default='lime', help='Explanation method')
    args = parser.parse_args()

    if args.Explanation == 'lime':
        lime=lime_libraray(args.data_path, args.model_name)
        #lime.build_for_UCI_Credit_Card()
        lime.build()

    elif args.Explanation == 'Anchors':
        anchor =anchors_libraray(args.data_path, args.model_name)
        anchor.anchor()

    elif args.Explanation == 'Shap':
        shap = shap_library(args.data_path,args.model_name)
        shap.build_for_FICO()