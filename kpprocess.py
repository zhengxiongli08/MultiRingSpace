
# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

# Get keypoints from differential pyramid

import taichi as ti
import numpy as np
import cv2 as cv
import pickle
import numba
from numba import njit, prange


# Functions
@ti.func
def get_max_value(img: ti.types.ndarray(), i: ti.i32, j: ti.i32) -> ti.i32:
    """
    Functions:
        获取当前图片内3*3范围内的极大值，主要针对上下两层
    Args:
        img: 图像
        i: 高度方向上的坐标
        j: 宽度方向上的坐标
    Returns:
        3*3区域内的极大值
    """
    result = img[i, j]
    for loc_1, loc_2 in ti.ndrange((i-1, i+2), (j-1, j+2)):
        value = img[loc_1, loc_2]
        if (value > result):
            result = value
    
    return result

@ti.func
def get_min_value(img: ti.types.ndarray(), i: ti.i32, j: ti.i32) -> ti.i32:
    """
    Functions:
        获取当前图片内3*3范围内的极小值，主要针对上下两层
    Args:
        img: 图像
        i: 高度方向上的坐标
        j: 宽度方向上的坐标
    Returns:
        3*3区域内的极小值
    """
    result = img[i, j]
    for loc_1, loc_2 in ti.ndrange((i-1, i+2), (j-1, j+2)):
        value = img[loc_1, loc_2]
        if (value < result):
            result = value
    
    return result

@ti.func
def is_max_value(img: ti.types.ndarray(), i: ti.i32, j: ti.i32) -> bool:
    """
    Functions:
        判断当前的[i, j]像素是否是3*3区域内的最大值，主要针对当前层
    Args:
        img: 图像
        i: 高度方向上的坐标
        j: 宽度方向上的坐标
    Returns:
        True表示是极大值，False表示不是极大值
    """
    cur_value = img[i, j]
    flag_1 = cur_value > img[i-1, j-1] and cur_value > img[i-1, j] and cur_value > img[i-1, j+1]
    flag_2 = cur_value > img[i, j-1] and cur_value > img[i, j+1]
    flag_3 = cur_value > img[i+1, j-1] and cur_value > img[i+1, j] and cur_value > img[i+1, j+1]
    result = flag_1 and flag_2 and flag_3
    
    return result

@ti.func
def is_min_value(img: ti.types.ndarray(), i: ti.i32, j: ti.i32) -> bool:
    """
    Functions:
        判断当前的[i, j]像素是否是3*3区域内的最小值，主要针对当前层
    Args:
        img: 图像
        i: 高度方向上的坐标
        j: 宽度方向上的坐标
    Returns:
        True表示是极小值，False表示不是极小值
    """
    cur_value = img[i, j]
    flag_1 = cur_value < img[i-1, j-1] and cur_value < img[i-1, j] and cur_value < img[i-1, j+1]
    flag_2 = cur_value < img[i, j-1] and cur_value < img[i, j+1]
    flag_3 = cur_value < img[i+1, j-1] and cur_value < img[i+1, j] and cur_value < img[i+1, j+1]
    result = flag_1 and flag_2 and flag_3
    
    return result

@ti.kernel
def _get_keypoints(img_self: ti.types.ndarray(), 
                 img_up: ti.types.ndarray(), 
                 img_low: ti.types.ndarray(), 
                 result: ti.types.ndarray(dtype=ti.i32)):
    """
    Functions:
        根据输入的三层图片，返回中间层图片的关键点位置，保存在一张ndarray中
        极大值的位置写入1，极小值的位置写入0
    Args:
        img_self: 中间层图片
        img_up: 上层图片
        img_lwo:下层图片
    Returns:
        None，利用传参时传入result的引用实现
    """
    img_height = img_self.shape[0]
    img_width = img_self.shape[1]
    
    for i, j in ti.ndrange((1, img_height), (1, img_width)):
        img_up_max = get_max_value(img_up, i, j)
        img_up_min = get_min_value(img_up, i, j)
        img_low_max = get_max_value(img_low, i, j)
        img_low_min = get_min_value(img_low, i, j)
        img_self_value = img_self[i, j]
        flag_max = is_max_value(img_self, i, j) and img_self_value > img_up_max and img_self_value > img_low_max
        flag_min = is_min_value(img_self, i, j) and img_self_value < img_up_min and img_self_value < img_low_min
        if (flag_max):
            result[i, j] = 1
        elif (flag_min):
            result[i, j] = -1
    
    return

def array2list(kp_map: ti.types.ndarray()) -> list:
    """
    Function:
        将Taichi函数生成的关键点ndarray转换为list处理
    Args:
        kp_map: 由Taichi函数生成的关键点ndarray
    Return:
        一个list，包含了许多关键点cv.KeyPoint对象
    """
    result = list()
    MAX_FLAG = 1
    MIN_FLAG = 0
    for i in range(0, kp_map.shape[0]):
        for j in range(0, kp_map.shape[1]):
            if (kp_map[i, j] == 1):
                kp = cv.KeyPoint(x=j, y=i, size=0, class_id=MAX_FLAG)
                result.append(kp)
            elif (kp_map[i, j] == -1):
                kp = cv.KeyPoint(x=j, y=i, size=0, class_id=MIN_FLAG)
                result.append(kp)
                
    return result

def get_keypoint_list(diff_list: list) -> list:
    """
    Functions:
        输入差分金字塔，输出可用层的关键点序列，并且放到一个list里面输出
    Args:
        diff_list: 差分金字塔
    Returns:
        关键点序列的序列
    """
    keypoint_list = list()
    diff_depth = len(diff_list)
    # 对除了底部和顶部的两张图片外的，开始找极值点
    for i in range(1, diff_depth - 1):
        # 获取上下和自己一共三张图像
        img_self = diff_list[i]
        img_up = diff_list[i + 1]
        img_low = diff_list[i - 1]
        keypoint_map = np.zeros_like(img_self, dtype=np.int32)
        _get_keypoints(img_self, img_up, img_low, keypoint_map)
        keypoint_set = array2list(keypoint_map)
        keypoint_list.append(keypoint_set)
        
    return keypoint_list

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
        loc_width = int(item.pt[0])
        loc_height = int(item.pt[1])
        minmax_flag = item.class_id       # 指示是最大值还是最小值的flag
        if (minmax_flag):       # 即为最大值
            cv.circle(result_img, (loc_width, loc_height), radius=7, color=RED, thickness=3)    # 画点为横纵坐标表示
        else:
            cv.circle(result_img, (loc_width, loc_height), radius=7, color=BLUE, thickness=3)
    
    return result_img

# Test case
if __name__ == "__main__":
    ti.init(arch = ti.cpu)
    with open("diff_list.pkl", "rb") as file:
        diff_list = pickle.load(file)
        get_keypoint_list(diff_list)
        