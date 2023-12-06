
# This program is used to get the eigen vectors for keypoints

import taichi as ti
import numba
import numpy as np
import pickle
from numba import njit, prange
# from conv import loc_conv


# Functions
@ti.func
def loc_conv(src: ti.types.ndarray(), 
             mask: ti.types.ndarray(), 
             coor_h: ti.i32, 
             coor_w: ti.i32) -> ti.f64:
    """
    Function:
        Do convolution in designated position
    Args:
        src: source matrix
        mask: convolution kernel
        i: convolution coordinates in height direction
        j: convolution coordinates in width direction
    Return:
        Convolution result
    """
    result = 0.0
    radius = int((mask.shape[0] - 1) / 2)
    size = mask.shape[0]
    for k, l in ti.ndrange((0, size), (0, size)):
        img_value = src[coor_h-radius+k, coor_w-radius+l]
        mask_value = mask[k, l]
        result += img_value * mask_value
        
    return result

def get_conv_eigen(img, kps, mask):
    """
    Warp for _get_conv_eigen.
    Get convolution values for all keypoints using a single kernel
    'result' is a numpy array whose length is keypoints' number
    """
    @ti.kernel
    def _get_conv_eigen(img: ti.types.ndarray(), 
                        kps: ti.types.ndarray(), 
                        mask: ti.types.ndarray(), 
                        result: ti.types.ndarray()):
        
        kps_length = kps.shape[0]
        for i in range(0, kps_length):
            coor_h, coor_w = kps[i, 0], kps[i, 1]
            result[i] = loc_conv(img, mask, coor_h, coor_w)
        
        return
    # Main part
    kps_length = kps.shape[0]
    result = np.zeros(kps_length)
    _get_conv_eigen(img, kps, mask, result)

    return result

def zscore_scale(mat_data):
    """
    Scale a matrix row by row using z-score
    """
    @njit
    def my_scale(vec_data):
        """
        Scale a vector using z-score.
        """
        # Get the average value
        mean_value = np.mean(vec_data)
        # Get the standard deviation value
        std_dev = np.std(vec_data)
        # Calcualte result
        result = (vec_data - mean_value) / std_dev
        
        return result
    # Main body
    result = np.zeros_like(mat_data)
    for i in range(0, mat_data.shape[0]):
        result[i] = my_scale(mat_data[i])
    
    return result

def get_conv_eigens(img, kps, mask_list):
    """
    Part 1 of eigens, energy of keypoints (convolution)
    After computing, normalize it
    """
    # Main part
    conv_pre = list()
    for mask in mask_list:
        temp = get_conv_eigen(img, kps, mask)
        conv_pre.append(temp)
        print(np.mean(temp))
    # np.vstack has a row number of mask_list's length
    # so, transpose it
    temp = np.vstack(conv_pre).T
    # Normalize it row by row using z-score
    result = zscore_scale(temp)
    
    return result

def get_diff_eigens(conv_eigens):
    """
    Part 2 of eigens, energy difference between keypoints' energy
    """
    # Generate a zero mat for results
    eigens_len = conv_eigens.shape[1]
    diff_len = int(eigens_len * (eigens_len - 1) / 2)
    result = np.zeros((conv_eigens.shape[0], diff_len))
    # Calculate difference
    count = 0
    for i in range(0, eigens_len):
        for j in range(i + 1, eigens_len):
            temp = conv_eigens[:, i] - conv_eigens[:, j]
            result[:, count] = temp
            count += 1
    # Normalize it row by row using z-score
    result = zscore_scale(result)
    
    return result
    
def get_eigens(img, kps, mask_list):
    """
    Get eigens vectors for all keypoints.
    Eigen vectors are a numpy array.
    Number of rows is the number of keypoints.
    Number of columns is the length of a single eigen vector.
    Vector composition should be carefully designed.
    
    Part 1 of eigens: 
    energy of keypoints (convolution)
    use convolution kernels in mask_list to calculate energy
    for every keypoint.
    Then scale it using z-score
    
    Part 2 of eigens:
    difference of energies
    
    """
    # Part 1, keypoints' energy
    conv_eigens = get_conv_eigens(img, kps, mask_list)
    
    # Part 2, keypoints' energy difference
    diff_eigens = get_diff_eigens(conv_eigens)
    
    eigens = (conv_eigens, diff_eigens)
    eigens = np.concatenate(eigens, axis=1)
    
    eigens = zscore_scale(eigens)
    
    return eigens
    
if __name__ == "__main__":
    ti.init(arch=ti.cpu)
    
    with open("./temp/eigen_var1.pkl", "rb") as file:
        img_origin_gray, kps, mask_list = pickle.load(file)

    eigens = get_eigens(img_origin_gray, kps, mask_list)
    print(f"Entire eigens shape: {eigens.shape}")
    
    print("Program finished!")
