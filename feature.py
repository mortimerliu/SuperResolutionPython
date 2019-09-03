#!/usr/bin/env python

import os
import random
import logging
import pandas as pd
import numpy as np
from utils import scan, read_img, get_feature_helper, get_label_helper

def feature(LR_dir, HR_dir, npoints = 1000):
    
    LR_paths = scan(LR_dir)
    HR_paths = scan(HR_dir)
    
    if not LR_paths.count() == HR_paths.count():
        raise Exception("LR and HR don't match")
    
    LR_arrs = pd.Series(LR_paths.map(read_img))
    HR_arrs = pd.Series(HR_paths.map(read_img))
    
    LR_HR_arrs = pd.DataFrame(zip(LR_arrs, HR_arrs))

    ###

    def feature_helper(LR_array3d, HR_array3d, npoints):

        #print('new_image') # debug

        np.random.seed(2019)

        xs = np.random.randint(LR_array3d.shape[0], size = npoints)
        ys = np.random.randint(LR_array3d.shape[1], size = npoints)

        coordinates = pd.DataFrame({'x': xs, 'y': ys})

        feat_mat = coordinates.apply(get_feature_helper, LR_array3d = LR_array3d, axis = 1)
        feat_mat = np.dstack(feat_mat).transpose(2, 0, 1)

        lab_mat = coordinates.apply(get_label_helper, LR_array3d = LR_array3d, HR_array3d = HR_array3d, axis = 1)
        lab_mat = np.dstack(lab_mat).transpose(2, 0, 1)

        return pd.Series({'feature': feat_mat, 'label': lab_mat})

    ###

    feat_lab = np.array(LR_HR_arrs.apply(lambda x: feature_helper(*x, npoints), axis = 1))
    
    feat_mat = np.vstack(feat_lab[:, 0])
    label_mat = np.vstack(feat_lab[:, 1])
    
    return feat_mat, label_mat


if __name__ == '__main__':
    
    LR_img_dir = '../data/train_set/LR'
    HR_img_dir = '../data/train_set/HR'
    npoints = 1000

    feat_mat, lab_mat = feature(LR_img_dir, HR_img_dir, npoints = npoints)

    np.save('../output/feat_mat.npy', feat_mat)
    np.save('../output/lab_mat.npy', lab_mat)



