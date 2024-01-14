
# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

# Get keypoints from differential pyramid

import numpy as np
import cv2 as cv
import numba
from numba import njit, prange


# Functions
@njit
def is_max_value(part):
    kp_r = int((part.shape[0] - 1) / 2)
    max_value = np.max(part)
    if (part[kp_r, kp_r] == max_value) and (np.count_nonzero(part == max_value) == 1):
        return True
    return False

@njit
def is_min_value(part):
    kp_r = int((part.shape[0] - 1) / 2)
    min_value = np.min(part)
    if (part[kp_r, kp_r] == min_value) and (np.count_nonzero(part == min_value) == 1):
        return True
    return False

@njit(parallel=True)
def _get_kps(img_up, img_self, img_down, kp_r) -> np.ndarray:
    """
    Get a keypoint map for 3 layes, up, down and self
    """
    img_height, img_width = img_self.shape[:2]
    result = np.zeros_like(img_self).astype(np.int32)
    # Traverse those pixels one by one
    for i in prange(kp_r, img_height+1-kp_r):
        for j in prange(kp_r, img_width+1-kp_r):
            # Extract 3 part with a size of 3x3
            img_up_part = img_up[i-kp_r:i+1+kp_r, j-kp_r:j+1+kp_r]
            img_down_part = img_down[i-kp_r:i+1+kp_r, j-kp_r:j+1+kp_r]
            img_self_part = img_self[i-kp_r:i+1+kp_r, j-kp_r:j+1+kp_r]
            # Get the max and min values of up level and down level
            img_up_max = np.max(img_up_part)
            img_up_min = np.min(img_up_part)
            img_down_max = np.max(img_down_part)
            img_down_min = np.min(img_down_part)
            # Determine whether the center pixel is max or min
            pixel = img_self[i, j]
            self_max = is_max_value(img_self_part)
            self_min = is_min_value(img_self_part)
            # Determine whether the center pixel is keypoint or not
            flag_max = self_max and (pixel > img_up_max) and (pixel > img_down_max)
            flag_min = self_min and (pixel < img_up_min) and (pixel < img_down_min)
            # Save results
            if (flag_max):
                result[i, j] = 1
            elif (flag_min):
                result[i, j] = -1
                
    return result

def map2array(kp_map):
    """
    Transform keypoints map into a numpy array
    First column is coordinate in height direction.
    Second column is coordinate in width direction.
    """
    coordinates = np.argwhere((kp_map == 1) | (kp_map == -1))

    return coordinates

def get_kps(diff_list, kp_r):
    """
    Get keypoint map from differential pyramid, 
    then transform it into a series of numpy array, 
    then connect them together and remove the duplicate rows
    """
    kps_list = list()
    diff_depth = len(diff_list)
    # Traverse differential pyramid
    for i in range(1, diff_depth - 1):
        # 3 layers, up, down, self
        img_self = diff_list[i]
        img_up = diff_list[i + 1]
        img_down = diff_list[i - 1]
        
        kps_map = _get_kps(img_self, img_up, img_down, kp_r)
        kps = map2array(kps_map)
        kps_list.append(kps)
    # Combine vertically
    temp = np.vstack(kps_list).astype(np.int32)
    # Remove duplicate rows
    result = np.unique(temp, axis=0)

    return result

def get_kps_from_mask(mask):
    """
    Generator keypoints from mask
    """
    # Generate the meshgrid
    height, width = mask.shape
    temp = np.zeros((height, width), dtype=np.int32)
    temp[::2, ::2] = 1
    # Filter wrong keypoints using mask
    kps_map = np.multiply(temp, mask)
    result = map2array(kps_map)
    
    return result

def get_color_keypoint_img(img: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    """
    Functions:
        Draw keypoints on the input image.
        Not distinguish between maximum and minimum values
    Args:
        keypoints: keypoints for a single level
        img: input image
    Returns:
        Image with keypoints on it
    """
    result_img = img.copy()
    GREEN = (0, 255, 0)
    for item in keypoints:
        coor_h, coor_w = item
        cv.circle(result_img, (coor_w, coor_h), radius=1, color=GREEN, thickness=2)
    
    return result_img

# Test case
if __name__ == "__main__":
    import pickle
    with open("./temp/diff_list.pkl", "rb") as file:
        diff_list = pickle.load(file)
    kps = get_kps(diff_list)
    image = cv.imread("../SlidesThumbnail/slide-2022-12-19T17-59-32-R5-S14.tiff")
    result = get_color_keypoint_img(kps, image)
    cv.imwrite("./kp.png", result)

        