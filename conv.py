
# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

# Get differential pyramid

import numba
import numpy as np
import taichi as ti
from numba import njit, prange


# Functions
@ti.func
def local_conv(src: ti.types.ndarray(), 
             mask: ti.types.ndarray(), 
             coor_h: ti.i32, 
             coor_w: ti.i32) -> ti.f32:
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
    result = ti.cast(0, ti.f32)
    radius = int((mask.shape[0] - 1) / 2)
    size = mask.shape[0]
    for k, l in ti.ndrange((0, size), (0, size)):
        img_value = src[coor_h-radius+k, coor_w-radius+l]
        mask_value = mask[k, l]
        result += img_value * mask_value
        
    return result

def convolve(img: np.ndarray, 
             mask: ti.ndarray) -> np.ndarray:
    """
    Function:
        Copy pixels at the boundaries, then send it to Taichi function
    Args:
        img: image matrix that is going to be convoluted
        mask: convolution kernel
    Return:
        Convolution results
    """
    @ti.kernel
    def _convolve(img_expand: ti.types.ndarray(), 
                mask: ti.types.ndarray(), 
                result: ti.types.ndarray()):
        """
        Function:
            Calculate the convolution result for an already expanded image
        Args:
            img_expand: the image matrix that has been expanded at the boundaries
            mask: the convolution kernel
            result: used to store results
        Return:
            None
        """
        radius = int((mask.shape[0] - 1) / 2)
        for i, j in ti.ndrange((0, result.shape[0]), (0, result.shape[1])):
            result[i, j] = local_conv(img_expand, mask, i+radius, j+radius)
        
        return
    # Main
    radius = int((mask.shape[0] - 1) / 2)
    pad_width = ((radius, radius), (radius, radius))
    img_expand = np.pad(img, pad_width, mode="edge")
    img_expand = img_expand.astype(np.float32)
    result = np.zeros_like(img, dtype=np.float32)
    # Rotate the convolution kernel
    temp = np.flipud(np.fliplr(mask))
    mask_rotate = np.ascontiguousarray(temp)
    _convolve(img_expand, mask_rotate, result)

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
    ti.init(arch=ti.gpu)
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
    mask_list = get_mask_list(5, 30, 7)
    img = np.random.randint(0, 255, (1000, 1000)).astype(np.uint8)
    conv_list = get_conv_list(img, mask_list)
    
    print("Program finished!")    
