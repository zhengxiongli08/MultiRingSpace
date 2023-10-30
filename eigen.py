
# This program is used to get the eigen vectors for keypoints

import taichi as ti
import numba
import numpy as np
import pickle
from numba import njit, prange
from conv import loc_conv


# Functions
@njit(parallel=True)
def new_loc_conv(src, mask, coor_h, coor_w):
    """
    Function:
        Do convolution in designated position
    Args:
        src: source matrix
        mask: convolution kernel
        coor_h: convolution coordinates in height direction
        coor_w: convolution coordinates in width direction
    Return:
        Convolution result
    """
    radius = int((mask.shape[0] - 1) / 2)
    size = mask.shape[0]
    result = 0
    for k in prange(0, size):
        for l in prange(0, size):
            src_value = src[coor_h-radius+k, coor_w-radius+l]
            mask_value = mask[k, l]
            result += src_value * mask_value

    return result

@njit(parallel=True)
def get_conv_eigen(img, kps, mask):
    """
    Get convolution values for all keypoints using a single kernel
    """
    kps_length = kps.shape[0]
    result = np.zeros(kps_length)
    for i in prange(0, kps_length):
        coor_h, coor_w = kps[i]
        result[i] = new_loc_conv(img, mask, coor_h, coor_w)
    
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

@njit
def get_conv_eigens(img, kps, mask_list):
    """
    Part 1 of eigens, energy of keypoints (convolution)
    """
    conv_pre = list()
    for mask in mask_list:
        temp = get_conv_eigen(img, kps, mask)
        conv_pre.append(temp)
        print(temp.shape)
    # np.vstack has a row number of mask_list's length
    # so, transpose it
    # result = np.vstack(conv_pre).T
    
    return
    

@njit(parallel=True)
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
    # eigens = np.zeros_like(conv)
    # # Normalize it
    # for i in prange(0, conv.shape[0]):
    #     eigens[i, :] = my_scale(conv[i, :])
    
    # return eigens
    
if __name__ == "__main__":
    # ti.init(arch=ti.gpu)
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
