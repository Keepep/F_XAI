import torch.nn as nn
import torch
from tqdm import tqdm
from utils import *
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge, lars_path
import numpy as np

class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()

        if torch.cuda_available():
            self.linear = nn.Linear(input_size, num_classes).cuda()
        else:
            self.linear = nn.Linear(input_size, num_classes)


    def forward(self, x):
        out = self.linear(x)
        return out


class lime_base():
    def __init__(self,kernel_fn,verbose, random_state):
        self.kernel_fn= kernel_fn
        self.verbose=verbose
        self.random_state=check_random_state(random_state)

    def build(self):
        model=LogisticRegression(self.mask_index.shape[1],1)
        optimizer=torch.optim.SGD(model.parameters(), lr=self.lr)
        for k in range(self.epoch):
            loss=0
            for i in range(self.mask_index.shape[0]):
                output=model.forward(self.mask_index[i])
                loss_tmp=torch.mul(self.dist[i].cuda(),\
                               torch.pow(torch.sub(self.os[i].cuda(),output),2))

                te=torch.pow(torch.sub(self.os[i].cuda(),output),2)
                loss=loss_tmp+loss

            all_linear1_params = torch.cat([x.view(-1) for x in model.linear.parameters()])
            l1_reg = self.l1_coeff * torch.norm(all_linear1_params, 1)
            loss=loss+l1_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print ('Epoch: [%d/%d], Loss: %.4f'
                   % (k + 1, self.epoch, loss.data[0]))
        for i in model.linear.parameters():
            weight=i
            break

        argsort=np.argsort(weight.data.cpu().numpy())

        k_index=argsort[0][-self.K:]
        print (weight)
        print (weight[0][k_index[0]])
        print (weight[0][k_index[9]])

        #k_index=argsort[0][:self.K]
        return k_index
    def feature_selection(self, data, labels, weights, num_features, method):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return self.forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':

            #because of alpha=0, l2 regularization is removed
            clf = Ridge(alpha=0, fit_intercept=True,
                        random_state=self.random_state)

            #sample_weight: Individual weights for each sample
            clf.fit(data, labels, sample_weight=weights)


            #clf.coef_ : weights of ridge classifier
            #data[0]: instance x
            #ordering feature and its value(contribution) for classification from largest
            #to smallest value

            feature_weights = sorted(zip(range(data.shape[0]),
                                         clf.coef_ * data[0]),
                                     key=lambda x: np.abs(x[1]),
                                     reverse=True)

            #return feature names ordering it from smallest to largest weight
            return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = self.generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return self.feature_selection(data, labels, weights,
                                          num_features, n_method)


    def explain_instance_with_data(self,
                                   neighborhood_data,
                                   neighborhood_labels,
                                   distances,
                                   label,
                                   num_features,
                                   feature_selection='auto',
                                   model_regressor=None):
        """Takes perturbed data, labels and distances, returns explanation.

        Args:
            neighborhood_data: perturbed data, 2d array. first element is
                               assumed to be the original data point.
            neighborhood_labels: corresponding perturbed labels. should have as
                                 many columns as the number of possible labels.
            distances: distances to original data point.
            label: label for which we want an explanation
            num_features: maximum number of features in explanation
            feature_selection: how to select num_features. options are:
                'forward_selection': iteratively add features to the model.
                    This is costly when num_features is high
                'highest_weights': selects the features that have the highest
                    product of absolute weight * original data point when
                    learning with all the features
                'lasso_path': chooses features based on the lasso
                    regularization path
                'none': uses all features, ignores num_features
                'auto': uses forward_selection if num_features <= 6, and
                    'highest_weights' otherwise.
            model_regressor: sklearn regressor to use in explanation.
                Defaults to Ridge regression if None. Must have
                model_regressor.coef_ and 'sample_weight' as a parameter
                to model_regressor.fit()

        Returns:
            (intercept, exp, score, local_pred):
            intercept is a float.
            exp is a sorted list of tuples, where each tuple (x,y) corresponds
            to the feature id (x) and the local weight (y). The list is sorted
            by decreasing absolute value of y.
            score is the R^2 value of the returned explanation
            local_pred is the prediction of the explanation model on the original instance
        """

        weights = self.kernel_fn(distances)
        labels_column = neighborhood_labels[:, label]
        used_features = self.feature_selection(neighborhood_data,
                                               labels_column,
                                               weights,
                                               num_features,
                                               feature_selection)

        if model_regressor is None:
            model_regressor = Ridge(alpha=0.08, fit_intercept=True,
                                    random_state=self.random_state)
        easy_model = model_regressor
        easy_model.fit(neighborhood_data[:, used_features],
                       labels_column, sample_weight=weights)

        prediction_score = easy_model.score(
            neighborhood_data[:, used_features],
            labels_column, sample_weight=weights)
        #local_pred means result about instance from a new simple model
        local_pred = easy_model.predict(neighborhood_data[0, used_features].reshape(1, -1))

        if self.verbose:
            print('Intercept', easy_model.intercept_)
            print('Prediction_local', local_pred,)
            print('Right:', neighborhood_labels[0, label])

        #easy_model.coef: contributions for features
        return (easy_model.intercept_,
                sorted(zip(used_features, easy_model.coef_),
                       key=lambda x: np.abs(x[1]), reverse=True),
                prediction_score, local_pred)