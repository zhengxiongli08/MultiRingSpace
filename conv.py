
# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

# Get differential pyramid

import numba
import numpy as np
from numba import njit, prange


# Functions
@njit(parallel=True)
def local_conv(src, mask, coor_h, coor_w):
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
    result = 0
    radius = int((mask.shape[0] - 1) / 2)
    size = mask.shape[0]
    for k in prange(0, size):
        for l in prange(0, size):
            img_value = src[coor_h-radius+k, coor_w-radius+l]
            mask_value = mask[k, l]
            result += img_value * mask_value
        
    return result

def convolve(img, mask):
    """
    Function:
        Copy pixels at the boundaries, then send it to Taichi function
    Args:
        img: image matrix that is going to be convoluted
        mask: convolution kernel
    Return:
        Convolution results
    """
    radius = int((mask.shape[0] - 1) / 2)
    pad_width = ((radius, radius), (radius, radius))
    img_expand = np.pad(img, pad_width, mode="edge")
    img_expand = img_expand.astype(np.float32)
    result = np.zeros_like(img, dtype=np.float32)
    # Rotate the convolution kernel
    temp = np.flipud(np.fliplr(mask))
    mask_rotate = np.ascontiguousarray(temp)
    for i in prange(0, img.shape[0]):
        for j in prange(0, img.shape[1]):
            result[i, j] = local_conv(img_expand, mask_rotate, i+radius, j+radius)
            
    return result

def build_ring(radius: int, thickness: int) -> np.ndarray:
    """
    Function:
        Create a ring-shape convolution kernel
    Args:
        radius: radius of kernel's inner circle
        thickness: thickness of the ring
    Returns:
        Convolution kernel matrix
    """
    width = 2 * (radius + thickness) + 1
    mask = np.zeros((width, width)).astype(np.float32)
    center = (width - 1) / 2
    count = 0
    # Calculate which pixel should be 1
    for i in range(0, width):
        for j in range(0, width):
            distance = np.sqrt((i-center)**2 + (j-center)**2)
            if (distance <= radius + thickness and distance > radius):
                mask[i,j] = 1
                count = count + 1
    mask = mask / count
    
    return mask

def get_mask_list(radius_min: int, 
                  radius_max: int, 
                  thickness: int) -> list:
    """
    Function:
        Get list of convolution list, 
        radius range from radius_min to radius_max,
        boundaries are included.
        Different ring has one third area overlapping.
    Args:
        radius_min: minimum radius
        radius_max: maximum radius
    Returns:
        List of convolution kernel
    """
    result_list = list()
    step = thickness // 3
    for i in range(radius_min, radius_max+1, step):
        mask = build_ring(i, thickness)
        result_list.append(mask)
    
    return result_list

def get_conv_list(img: np.ndarray, 
                  mask_list: list) -> list:
    """
    Function:
        Get convolution image pyramid
    Args:
        img: image after preprocessing
        mask_list: list of convolution kernels
    Returns:
        List of convolution results
    """
    conv_list = list()
    for mask in mask_list:
        img_conv = convolve(img, mask)
        img_conv = img_conv.astype(np.uint8)
        conv_list.append(img_conv)
        
    return conv_list

def get_diff_list(conv_list: list) -> list:
    """
    Functions:
        Get differential pyramid from convolution pyramid
    Args:
        conv_list: list of convolution results
    Returns:
        list of differntial results
    """
    diff_list = list()
    depth = len(conv_list) - 1
    for i in range(0, depth):
        img_diff = conv_list[i + 1] - conv_list[i]
        diff_list.append(img_diff)
        
    return diff_list

if __name__ == "__main__":
    src_mat = np.arange(25).reshape((5, 5))
    print(src_mat)
    kernel = np.ones((3, 3))
    kernel[0, 0] = 0
    res1 = convolve(src_mat, kernel)
    print(res1)
    
    print("Program finished!")    
