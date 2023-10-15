
# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

# 由关键点序列和差分图构建特征向量（特征描述子部分）

import numpy as np
import cv2 as cv
import taichi as ti
from conv import loc_conv


# Functions
def amalgamate_kp(keypoint_list: list) -> set:
    """
    Functions:
        合并由差分图得出的关键点序列
    Args:
        keypoint_list: 差分图的关键点序列
    Returns:
        关键点序列的并集
    """
    result = set()
    for keypoints in keypoint_list:
        keypoints = set(keypoints)
        result.update(keypoints)
    
    result = tuple(result)
    
    return result

def keypoints_transform(keypoints: set) -> np.ndarray:
    """
    Function:
        将关键点并集转换为一个nunpy数组，每一行都是一个关键点的信息
    """
    result = np.zeros((len(keypoints), 2), dtype=np.int32)
    for i in range(0, len(keypoints)):
        result[i] = keypoints[i].pt
        
    return result

def generate_des_mat(masks: list, keypoints: np.ndarray, diff_list: list) -> np.ndarray:
    """
    Function:
        根据卷积核序列信息，关键点个数和差分金字塔层数生成空白的卷积核矩阵
        矩阵的行数即为关键点的个数
        矩阵的列数：
        1. 对于原始图像，都需要对关键点位置做卷积求出能量序列，所以有卷积核数量个能量特征
        2. 对于原始图像，都需要对关键点位置做梯度求出梯度序列，此处为排列组合数，为n*(n-1)/2
        3. 对于每张差分图像，都需要对关键点位置做卷积求出能量序列，所以有卷积核数量个能量特征
        4. 对于每张差分图像，都需要对关键点位置做卷积求出梯度序列，所以有n*(n-1)/2个能量特征
        5. 由于有多张差分图像，所以第3项和第4项的数量要乘上差分图像个数
    Args:
        masks: 卷积核序列
        keypoints: 关键点并集
        diff_list: 差分金字塔
    Return:
        生成的空白特征描述子矩阵
    """
    diff_list_len = diff_list.shape[0]
    masks_len = len(masks)
    keypoints_num = keypoints.shape[0]
    pass

@ti.func
def myScale(array: ti.types.ndarray()):
    """
    Function:
        对输入的array进行z-score归一化处理，原地处理
    """
    sum = ti.f64(0.0)
    for i in range(0, array.shape[0]):
        sum += array[i]
        
    for i in range(0, array.shape[0]):
        array[i] = array[i] / sum
        
    return

@ti.kernel
def get_kp_eigen_origin(img_expand: ti.types.ndarray(), mask: ti.types.ndarray(), keypoints: ti.types.ndarray(), result: ti.types.ndarray()):
    """
    Function:
        对于给定的mask，计算所有关键点的卷积能量值
    """
    for i in range(0, keypoints.shape[0]):
        loc_h = keypoints[i, 0]
        loc_w = keypoints[i, 1]
        result[i] = loc_conv(img_expand, mask, loc_h, loc_w)
    
    myScale(result)
    
    return

def get_eigen(img_expand: np.ndarray, diff_list: list, keypoint_list: list, masks: list):
    """
    Function:
        计算特征向量矩阵
    """
    # 获取关键点并集
    keypoints = amalgamate_kp(keypoint_list)
    keypoints_array = keypoints_transform(keypoints)
    eigens = np.zeros((len(keypoints), len(masks)), dtype=np.float32)
    for i in range(len(masks)):
        mask = masks[i]
        temp = np.zeros(len(keypoints)).astype(np.float64)
        get_kp_eigen_origin(img_expand, mask, keypoints_array, temp)
        eigens[:, i] = temp
    
    return keypoints, eigens

if __name__ == "__main__":
    ti.init(arch = ti.cpu)
    img = np.arange(100).reshape(10, 10)
    mask = np.ones((3, 3))
    k1 = (1, 1)
    k2 = (2, 2)
    k3 = (2, 3)
    kps = np.array([k1, k2, k3])
    result = np.zeros(3)
    get_kp_eigen_origin(img, mask, kps, result)
    print(img)
    print(result)