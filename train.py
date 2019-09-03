#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV


def grid_search(lab, feat, tuned_parameters):
    
    # print('start a new cv')
    
    gbm = GradientBoostingRegressor(random_state = 2019) 
    
    clf = GridSearchCV(gbm, param_grid = tuned_parameters, 
                       scoring = 'neg_mean_squared_error',
                       n_jobs = 4, 
                       iid = False, 
                       cv = 5)
    clf.fit(feat, lab)
    
    print(clf.grid_scores_)
    
    return(clf.best_params_, clf.best_score_)


def train(feat_mat, lab_mat, param_grid:

    gs_results = []

    for k in range(3):

        print('channel %d' % k)

        feat = feat_mat[:,:,k]
        lab = lab_mat[:,:,k]
        lab = pd.DataFrame(lab)

        gs_result = lab.apply(grid_search, feat = feat, 
                              tuned_parameters = param_grid, axis = 0)
        gs_results.append(gs_result)

    return gs_results


if __name__ == '__main__':

    feat_mat = np.load('../output/feat_mat_new.npy')
    lab_mat = np.load('../output/lab_mat_new.npy')

    tuned_parameters_set_1 = {'learning_rate': [0.15, 0.1, 0.05, 0.01],
                              'n_estimators': [300, 500, 600, 700]}
    tuned_parameters_set_2 = {'max_depth': [3, 4, 5, 6],
                              'subsample': [0.6, 0.65, 0.7, 0.75, 0.8]}


    gbm = GradientBoostingRegressor(loss = 'ls',
                                    learning_rate = 0.05, 
                                    n_estimators = 500, 
                                    subsample = 0.8, 
                                    max_depth = 4, 
                                    random_state = 2019)

    model_list = []
    for i in range(12):
        X = feat_mat[:, :, i // 4]
        y = lab_mat[:, i % 4, i // 4]
        model = gbm.fit(X, y)
        model_list.append(model)
    
    import pickle
    with open('../output/model_list.pkl', 'wb') as fh:
        pickle.dump(model_list, fh)
        