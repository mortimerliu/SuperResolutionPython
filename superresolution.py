#!/usr/bin/env python

import os
import pickle
import random
import logging
import pandas as pd
import numpy as np
from PIL import Image
from utils import scan, read_img, path_leaf, reshape_helper, get_feature_helper


def super_resolution(LR_dir, model_list, out_dir):
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    LR_paths = scan(LR_dir)
    LR_leaves = LR_paths.map(path_leaf)
    out_paths = LR_leaves.map(lambda x: os.path.join(out_dir, x))

    ###
    
    def super_resolution_(LR_path, model_list, out_dir):
        
        LR_arr3d = read_img(LR_path)
        nrow = LR_arr3d.shape[0]
        ncol = LR_arr3d.shape[1]
        
        xs = np.repeat(list(range(nrow)), ncol)
        ys = np.array(list(range(ncol)) * nrow)
        coordinates = pd.DataFrame({'x':xs, 'y':ys})
        
        feat_mat = coordinates.apply(get_feature_helper, LR_array3d = LR_arr3d, axis = 1)
        feat_mat = np.dstack(feat_mat).transpose(2, 0, 1) # shape = (nrow * ncol, 8, 3)
        
        pred_mat = np.empty((nrow * ncol, 4, 3))
        
        for i in range(12):
            pred_mat[:, i % 4, i // 4] =  model_list[i].predict(feat_mat[:, :, i // 4])
            
        HR_array3d = reshape_helper(nrow, ncol, pred_mat)
        LR_arr3d_resize = np.apply_along_axis(np.repeat, 1, LR_arr3d, 2)
        LR_arr3d_resize = np.apply_along_axis(np.repeat, 0, LR_arr3d_resize, 2)
        HR_array3d = HR_array3d + LR_arr3d_resize

        HR_array3d = HR_array3d.astype('uint8')
        
        HR_img = Image.fromarray(HR_array3d)
        HR_img.save(out_dir)

    ###
    
    HR_LR_paths = pd.DataFrame({'path': out_paths, 'LR_paths': LR_paths})
    HR_LR_paths.apply(lambda x: super_resolution_(x[1], model_list, x[0]), axis = 1)


if __name__ == '__main__':
    
    LR_img_dir = '../data/test_set/LR_test'
    model_path = '../output/model_list.pkl'
    our_dir = '../data/test_set/HR_pred_test'

    with open('../output/model_list.pkl', 'rb') as fh:
        model_list = pickle.load(fh)

    super_resolution(LR_img_dir, model_list, our_dir)

