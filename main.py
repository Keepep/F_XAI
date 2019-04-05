
import argparse
from Explanation.LIME import *
import sklearn.datasets
import lime
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classifier for financial data')
    parser.add_argument('--data_path', type=str, default='Classifier/heloc_dataset_v1.csv', help='Dataset path')
    parser.add_argument('--model_name', type=str, default='Classifier/trained_model/Random_Forest.sav', help='Model Name')

    args = parser.parse_args()
    iris = sklearn.datasets.load_iris()

    train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(iris.data, iris.target,
                                                                                      train_size=0.80)
    print type(test[1])
    print test[1]
    print np.shape(test[1])
    lime=lime_libraray(args.data_path, args.model_name)
    lime.tabular()


