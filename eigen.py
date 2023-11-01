
# This program is used to get the eigen vectors for keypoints

import taichi as ti
import numba
import numpy as np
import pickle
from numba import njit, prange
from conv import loc_conv


# Functions
@ti.kernel
def _get_conv_eigen(img: ti.types.ndarray(), 
                   kps: ti.types.ndarray(), 
                   mask: ti.types.ndarray(), 
                   result: ti.types.ndarray()):
    """
    Get convolution values for all keypoints using a single kernel
    'result' is a numpy array whose length is keypoints' number
    """
    kps_length = kps.shape[0]
    for i in range(0, kps_length):
        coor_h, coor_w = kps[i, 0], kps[i, 1]
        result[i] = loc_conv(img, mask, coor_h, coor_w)
    
    return

def get_conv_eigen(img, kps, mask):
    """
    Warp for _get_conv_eigen
    """
    kps_length = kps.shape[0]
    result = np.zeros(kps_length)
    _get_conv_eigen(img, kps, mask, result)
    
    return result

@njit
def my_scale(data):
    """
    Scale the eigens vectors using z-score.
    """
    # Get the average value
    mean = np.mean(data)
    # Get the standard deviation value
    std_dev = np.std(data)
    # Calcualte result
    result = (data - mean) / std_dev
    
    return result

def get_conv_eigens(img, kps, mask_list):
    """
    Part 1 of eigens, energy of keypoints (convolution)
    Do not need to normalize it
    """
    conv_pre = list()
    for mask in mask_list:
        temp = get_conv_eigen(img, kps, mask)
        conv_pre.append(temp)
    # np.vstack has a row number of mask_list's length
    # so, transpose it
    result = np.vstack(conv_pre).T
    
    return result
    
def get_eigens(img, kps, mask_list):
    """
    Get eigens vectors for all keypoints.
    Eigen vectors are a numpy array.
    Number of rows is the number of keypoints.
    Number of columns is the length of a single eigen vector.
    Vector composition should be carefully designed.
    """
    # Part 1, keypoints' energy
    conv = get_conv_eigens(img, kps, mask_list)
    eigens = np.zeros_like(conv)
    # Normalize it
    for i in prange(0, conv.shape[0]):
        eigens[i, :] = my_scale(conv[i, :])
    
    return eigens
    
if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    with open("./temp/diff_list.pkl", "rb") as file:
        diff_list = pickle.load(file)
    with open("./temp/img_nobg_gray.pkl", "rb") as file:
        img_nobg_gray = pickle.load(file)
    with open("./temp/kp_list.pkl", "rb") as file:
        kps = pickle.load(file)
    with open("./temp/mask_list.pkl", "rb") as file:
        mask_list = pickle.load(file)

    eigens = get_eigens(img_nobg_gray, kps, mask_list)
    print(eigens.shape)

    pass
