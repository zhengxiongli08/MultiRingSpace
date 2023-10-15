
# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

import numpy as np
import numba as nb
import cv2 as cv
from scipy.spatial.distance import cdist

# Functions declaration
# ---------------------纯调用OpenCV库函数组成的匹配方法---------------------------
def bf_match(eigens_1: np.ndarray, eigens_2: np.ndarray, threshold) -> list:
    """
    Function:
        进行暴力匹配，并且通过knn在距离矩阵中给每一行都选出两个最小的距离值，然后通过阈值比较的方法选出更优的一个
    Args:
        eigens_1: 第一张图的特征向量矩阵
        eigens_2: 第二张图的特征向量矩阵
    Return:
        匹配对，为cv.Dmatch类型
    """
    bf = cv.BFMatcher()
    matches = bf.knnMatch(eigens_1, eigens_2, k=2)
    
    good_matches = list()
    for m, n in matches:
        if m.distance < threshold*n.distance:
            good_matches.append([m])
            
    return good_matches

def homo_match(keypoints_1: tuple, keypoints_2: tuple, eigens_1: np.ndarray, eigens_2: np.ndarray, threshold: float=0.7) -> list:
    """
    Function:
        对粗匹配结果进行筛选，通过RANSAC算法得到一个精匹配结果
    Args:
        keypoints_1: 第一张图的关键点
        keypoints_2: 第二张图的关键点
        eigens_1: 第一张图的特征向量矩阵
        eigens_2: 第二张图的特征向量矩阵
        threshold: 暴力匹配筛选用的阈值
    Return:
        匹配对
    """
    good_matches = bf_match(eigens_1, eigens_2, threshold)
    src_pts = np.float32([match.queryIdx for match in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([match.trainIdx for match in good_matches]).reshape(-1, 1, 2)
    _, mask = cv.findHomography(src_pts,dst_pts,cv.RANSAC, 5)
    match_mask = mask.ravel().tolist()
    
    return good_matches, match_mask
    
if __name__ == '__main__':
    e1 = np.random.rand(10, 2)
    e2 = np.random.rand(15, 2)
    e1 = e1.astype(np.float32)
    e2 = e2.astype(np.float32)
    result = bf_match(e1, e2, 0.7)
    print(result)