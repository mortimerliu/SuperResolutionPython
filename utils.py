import os
import random
import logging
import pandas as pd
import numpy as np
from PIL import Image


def scan(img_dir):
    with os.scandir(img_dir) as it:
        img_paths_list = [os.path.join(img_dir, entry.name) for entry in it if not entry.name.startswith('.') and entry.is_file()]
    img_paths_list.sort()
    img_paths = pd.Series(img_paths_list)
    return(img_paths)


def read_img(path):
    arr = np.asarray(Image.open(path))
    return arr


def pad_zeros(array3d):
    array3d0 = np.zeros(shape = (array3d.shape[0] + 2, array3d.shape[1] + 2, array3d.shape[2]))
    for k in range(array3d.shape[2]):   
        array3d0[1: array3d.shape[0] + 1, 1: array3d.shape[1] + 1, k] = array3d[:,:,k]
    return array3d0


def path_leaf(path):
    head, tail = os.path.split(path)
    return tail or os.path.basename(head)


def get_feature_helper(coordinate, LR_array3d):

    '''
    helper function to get neighbors (0-7) of a single pixel point (x)
    -------------
    | 0 | 1 | 2 |
    -------------
    | 3 | x | 4 |
    -------------
    | 5 | 6 | 7 |
    -------------
    '''

    feat = np.empty(shape = (8, 3))

    LR_array3d0 = pad_zeros(LR_array3d)

    x = coordinate[0] + 1
    y = coordinate[1] + 1
    
    for k in range(3):

        LR_array2d = LR_array3d0[:, :, k]

        feat[0, k] = LR_array2d[x - 1, y - 1] - LR_array2d[x, y]
        feat[1, k] = LR_array2d[x,     y - 1] - LR_array2d[x, y]
        feat[2, k] = LR_array2d[x + 1, y - 1] - LR_array2d[x, y]
        feat[3, k] = LR_array2d[x - 1, y    ] - LR_array2d[x, y]
        feat[4, k] = LR_array2d[x + 1, y    ] - LR_array2d[x, y]
        feat[5, k] = LR_array2d[x - 1, y + 1] - LR_array2d[x, y]
        feat[6, k] = LR_array2d[x,     y + 1] - LR_array2d[x, y]
        feat[7, k] = LR_array2d[x + 1, y + 1] - LR_array2d[x, y]

    return feat


def get_label_helper(coordinate, LR_array3d, HR_array3d):

    '''
    helper function to get labels (0-4) of a single pixel point (x)
    ---------
    | 0 | 1 |
    ---------
    | 2 | 3 |
    ---------
    '''

    lab = np.empty(shape = (4, 3))

    x = coordinate[0]
    y = coordinate[1]
    
    for k in range(3):
        
        LR_array2d = LR_array3d[:, :, k]
        HR_array2d = HR_array3d[:, :, k]

        lab[0, k] = HR_array2d[2 * x    , 2 * y    ] - LR_array2d[x, y]
        lab[1, k] = HR_array2d[2 * x + 1, 2 * y    ] - LR_array2d[x, y]
        lab[2, k] = HR_array2d[2 * x    , 2 * y + 1] - LR_array2d[x, y]
        lab[3, k] = HR_array2d[2 * x + 1, 2 * y + 1] - LR_array2d[x, y]

    return lab


def reshape_helper(nrow, ncol, pred_array3d):
    
    HR_array3d = np.empty(shape = (nrow * 2, ncol * 2, 3))
    
    for k in range(3):
        pred_array2d = pred_array3d[:, :, k]
        HR_array2d = pd.DataFrame(pred_array2d).apply(lambda x: np.array(x).reshape(2, 2), axis = 1)
        HR_array2d = np.hstack(HR_array2d).reshape(nrow * 2, ncol * 2)
        HR_array2d.sort(axis = 0)
        HR_array3d[:, :, k] = HR_array2d
        
    return(HR_array3d)
