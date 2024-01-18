
# This program is used to get the eigen vectors for keypoints

import sys
import numpy as np
import pickle
import multiprocessing
from numba import njit, prange
import time


# Functions
@njit
def local_conv(img, kernel, coor_h, coor_w):
    """
    Function:
        Do convolution in designated position
    Args:
        src: source matrix
        kernel: convolution kernel
        i: convolution coordinates in height direction
        j: convolution coordinates in width direction
    Return:
        Convolution result
    """
    # Calculate the convolution result
    radius = int((kernel.shape[0] - 1) / 2)
    img_part = img[coor_h-radius:coor_h+radius+1, coor_w-radius:coor_w+radius+1]
    temp = np.ravel(img_part * kernel)
    overall = np.mean(temp)
    # Divide the dot product results by the convolution result
    temp1 = temp[temp > overall]
    temp2 = temp[(temp <= overall) & (temp > 0)]
    # Determine whether the part is empty or not
    if temp1.shape[0] != 0:
        greater_part = np.mean(temp1)
    else:
        greater_part = 0
    
    if temp2.shape[0] != 0:
        less_part = np.mean(temp2)
    else:
        less_part = 0
    
    result = np.array((overall, greater_part, less_part))

    return result

def get_conv_eigen(img, kps, kernel):
    """
    Warp for _get_conv_eigen
    Because numba doesn't support np.pad operation
    """
    @njit(parallel=True)
    def _get_conv_eigen(img, kernel, kps):
        """
        Get convolution values for all keypoints using a single kernel
        'result' is a numpy array whose length is keypoints' number
        """
        # Main part
        kps_length = kps.shape[0]
        result = np.zeros((kps_length, 3))
        for i in prange(0, kps_length):
            coor_h, coor_w = kps[i, 0], kps[i, 1]
            result[i] = local_conv(img, kernel, coor_h, coor_w)

        return result
    # Considering some keypoints are near to border
    # Paddle the image and shift the keypoints
    radius = int((kernel.shape[0] - 1) / 2)
    pad_width = ((radius, radius), (radius, radius))
    img_expand = np.pad(img, pad_width, mode="edge")
    kps = kps + radius
    # Calculate eigenvectors from convolution
    result = _get_conv_eigen(img_expand, kernel, kps)
    
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

def get_conv_eigens(img, kps, kernel_list):
    """
    Part 1 of eigens, energy of keypoints (convolution)
    After computing, normalize it
    """
    # Prepare the input parameters
    jobs = [(img, kps, kernel) for kernel in kernel_list]
    with multiprocessing.Pool() as pool:
        conv_pre = pool.starmap(get_conv_eigen, jobs)
    temp = np.hstack(conv_pre)
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
    
def get_eigens(img, kps, kernel_list):
    """
    Get eigens vectors for all keypoints.
    Eigen vectors are a numpy array
    Number of rows is the number of keypoints.
    Number of columns is the length of a single eigen vector.
    Vector composition should be carefully designed.
    
    Part 1 of eigens: 
    energy of keypoints (convolution)
    use convolution kernels in kernel_list to calculate energy
    for every keypoint.
    Then scale it using z-score
    
    Part 2 of eigens:
    difference of energies
    
    """
    # Part 1, keypoints' energy
    conv_eigens = get_conv_eigens(img, kps, kernel_list)

    # Part 2, keypoints' energy difference
    diff_eigens = get_diff_eigens(conv_eigens)

    eigens = (conv_eigens, diff_eigens)
    eigens = np.concatenate(eigens, axis=1)

    eigens = zscore_scale(eigens)
    
    return eigens
    
if __name__ == "__main__":    
    with open("./data.pkl", "rb") as file:
        img_mean, kps, kernel_list = pickle.load(file)

    a1 = time.time()
    res = get_eigens(img_mean, kps, kernel_list)
    a2 = time.time()
    
    print(f"Eigen calculation cost: {a2-a1:.3f}s")
    print("Program finished!")
