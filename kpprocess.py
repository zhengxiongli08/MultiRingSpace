
# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

# Get keypoints from differential pyramid

import numpy as np
import cv2 as cv
import numba
from numba import njit, prange


# Functions
def map2list(kp_map):
    """
    Transform keypoints map into a list
    """
    max_coordinates_list = list(zip(*np.where(kp_map == 1)))
    min_coordinates_list = list(zip(*np.where(kp_map == -1)))
    kps_list = list()
    for item in max_coordinates_list:
        # coor_h, coor_w = item
        coor_h = int(item[0])
        coor_w = int(item[1])
        kp = cv.KeyPoint(x=coor_w, y=coor_h, size=0, class_id=1)
        kps_list.append(kp)
    
    for item in min_coordinates_list:
        # coor_h, coor_w = item
        coor_h = int(item[0])
        coor_w = int(item[1])
        kp = cv.KeyPoint(x=coor_w, y=coor_h, size=0, class_id=0)
        kps_list.append(kp)
    
    return kps_list

def map2array(kp_map):
    """
    Transform keypoints map into a numpy array
    """
    coordinates = np.argwhere((kp_map == 1) | (kp_map == -1))

    return coordinates

@njit
def is_max_value(part):
    max_value = np.max(part)
    if (part[1, 1] == max_value) and (np.count_nonzero(part == max_value) == 1):
        return True
    return False

@njit
def is_min_value(part):
    min_value = np.min(part)
    if (part[1, 1] == min_value) and (np.count_nonzero(part == min_value) == 1):
        return True
    return False

@njit(parallel=True)
def _get_kp(img_up, img_self, img_down) -> np.ndarray:
    """
    Get a keypoint map for 3 layes, up, down and self
    """
    img_height, img_width = img_self.shape[:2]
    result = np.zeros_like(img_self).astype(np.int32)
    for i in prange(1, img_height):
        for j in prange(1, img_width):
            img_up_part = img_up[i-1:i+2, j-1:j+2]
            img_down_part = img_down[i-1:i+2, j-1:j+2]
            img_self_part = img_self[i-1:i+2, j-1:j+2]
            img_up_max = np.max(img_up_part)
            img_up_min = np.min(img_up_part)
            img_down_max = np.max(img_down_part)
            img_down_min = np.min(img_down_part)
            
            pixel = img_self[i, j]
            # self_max = (pixel == np.max(img_self_part))
            # self_min = (pixel == np.min(img_self_part))
            self_max = is_max_value(img_self_part)
            self_min = is_min_value(img_self_part)
            
            flag_max = self_max and (pixel > img_up_max) and (pixel > img_down_max)
            flag_min = self_min and (pixel < img_up_min) and (pixel < img_down_min)

            if (flag_max):
                result[i, j] = 1
            elif (flag_min):
                result[i, j] = -1
                
    return result

def get_keypoint_list(diff_list):
    """
    Get keypoint map list from differential pyramid
    """
    keypoint_list = list()
    diff_depth = len(diff_list)
    # Traverse differential pyramid
    for i in range(1, diff_depth - 1):
        # 3 layers, up, down, self
        img_self = diff_list[i]
        img_up = diff_list[i + 1]
        img_down = diff_list[i - 1]
        
        kps = _get_kp(img_self, img_up, img_down)
        kps = map2list(kps)
        keypoint_list.append(kps)
    
    return keypoint_list

def get_keypoint_array(diff_list):
    """
    Get keypoint map list from differential pyramid
    """
    keypoint_list = list()
    diff_depth = len(diff_list)
    # Traverse differential pyramid
    for i in range(1, diff_depth - 1):
        # 3 layers, up, down, self
        img_self = diff_list[i]
        img_up = diff_list[i + 1]
        img_down = diff_list[i - 1]
        
        kps = _get_kp(img_self, img_up, img_down)
        kps = map2array(kps)
        keypoint_list.append(kps)
    
    result = np.vstack(keypoint_list)

    return result

def get_color_keypoint_img(keypoints: list, img: np.ndarray) -> np.ndarray:
    """
    Functions:
        根据关键点序列，对原图片进行上色，极大值点为红色，极小值点为蓝色
    Args:
        keypoints: 单幅图的关键点序列
        img: 原始图像
    Returns:
        上色后的图像
    """
    result_img = img.copy()
    RED = (0, 0, 255)
    BLUE = (255, 0, 0)
    for item in keypoints:
        coor_w = int(item.pt[0])
        coor_h = int(item.pt[1])
        minmax_flag = item.class_id       # 指示是最大值还是最小值的flag
        if (minmax_flag):       # 即为最大值
            cv.circle(result_img, (coor_w, coor_h), radius=2, color=RED, thickness=2)    # 画点为横纵坐标表示
        else:
            cv.circle(result_img, (coor_w, coor_h), radius=2, color=BLUE, thickness=2)
    
    return result_img

# Test case
if __name__ == "__main__":
    import pickle
    with open("./diff_list.pkl", "rb") as file:
        diff_list = pickle.load(file)
        print(diff_list[0].shape)
        a = get_keypoint_array(diff_list)
        # image = cv.imread("../SlidesThumbnail/slide-2022-12-19T17-59-32-R5-S14.tiff")
        # count = 0
        # for i in a:
        #     count += 1
        #     kps_list = map2list(i)
        #     print(len(kps_list))
        #     new_img = get_color_keypoint_img(kps_list, image)
        #     cv.imwrite(f"../result/{count}.png", new_img)
        print(a.shape)


    pass
        