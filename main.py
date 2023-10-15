
# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

# Multiple scale ring space

# Inner Libs
import os
import cv2 as cv
import numpy as np
from datetime import datetime
import sys
# Self Libs
import preprocess
import conv
import kpprocess
import eigenprocess
import match
import taichi as ti

ti.init(arch = ti.cpu)

# Functions
def compute(img_original, target_path):
    """
    对输入的图像进行计算，给出关键点的坐标和特征向量
    """
    # 读取图片，并作预处理（去除背景，填充空洞）
    img_preprocessed = preprocess.remove_background2(img_original)
    cv.imwrite(target_path + "img_preprocessed.png", img_preprocessed)
    print("Preprocess completed successfully!")
    
    # 获取卷积核序列
    mask_list = conv.get_mask_list(RADIUS_MIN, RADIUS_MAX, THICKNESS)
    print("Get mask list successfully!")
    
    # 获取卷积金字塔
    conv_list, img_expand = conv.get_conv_list(img_preprocessed, mask_list)
    for i in range(0, len(conv_list)):
        img = conv_list[i]
        cv.imwrite(target_path + "img_conv_{0}.png".format(i), img)
    print("Get convolve list successfully!")
    
    # 获取差分金字塔
    diff_list = conv.get_diff_list(conv_list)
    for i in range(0, len(diff_list)):
        img = diff_list[i]
        img_inverse = preprocess.inverse(img)
        img_equal = cv.equalizeHist(img_inverse)
        img_colored = cv.applyColorMap(img_equal, cv.COLORMAP_WINTER)
        cv.imwrite(target_path + "img_diff_color_{}.png".format(i), img_colored)
    print("Get differential list successfully!")
    
    # 获取关键点
    keypoint_list = kpprocess.get_keypoint_list(diff_list)
    for i in range(0, len(keypoint_list)):
        print("Diff img{0} keypoints number:".format(i), len(keypoint_list[i]))
        img_color = kpprocess.get_color_keypoint_img(keypoint_list[i], img_original)
        cv.imwrite(target_path + "img_keypoint_color_{0}.png".format(i), img_color)
    print("Get keypoint list successfully!")
    
    # 特征向量描述
    keypoints, eigens = eigenprocess.get_eigen(img_expand, diff_list, keypoint_list, mask_list)
    print("Get eigen vector successfully! \n")
    
    return keypoints, eigens
    
# Main function
if __name__ == "__main__":
    # Parameters
    IMG_PATH_1 = "/mnt/Disk1/whole slide image analysis/Lizhengxiong/manual_thumbnail/biopsy_2_1.png"
    IMG_PATH_2 = "/mnt/Disk1/whole slide image analysis/Lizhengxiong/manual_thumbnail/biopsy_2_4.png"
    RADIUS_MIN = 10
    RADIUS_MAX = 20
    THICKNESS = 7
    
    # 创建结果文件夹
    # 获取当前日期和时间
    current_time = datetime.now()
    # 格式化日期和时间为字符串
    time_str = current_time.strftime("%Y%m%d_%H%M%S")
    # 创建文件夹
    current_dir = os.getcwd()
    parent_dir = os.path.dirname(current_dir)
    folder_path = os.path.join(parent_dir, time_str)
    TARGET_PATH_1 = folder_path + "/result_figs_1/"
    TARGET_PATH_2 = folder_path + "/result_figs_2/"
    if not os.path.isdir(folder_path):
        os.makedirs(TARGET_PATH_1)
        os.makedirs(TARGET_PATH_2)
        
    # 重定向输出
    file = open(folder_path + "/log.txt", "w")
    sys.stdout = file
    
    # 读取图像
    img_1 = cv.imread(IMG_PATH_1)
    img_2 = cv.imread(IMG_PATH_2)
    # 获取关键点位置和特征向量
    keypoints_1, eigens_1 = compute(img_1, TARGET_PATH_1)
    keypoints_2, eigens_2 = compute(img_2, TARGET_PATH_2)
    # 找到匹配对
    good_matches = match.bf_match(eigens_1, eigens_2, threshold=0.7)
    
    # 画出结果
    img_match = cv.drawMatchesKnn(img_1, keypoints_1, img_2, keypoints_2, good_matches, None, flags=2)
    
    cv.imwrite(folder_path + "/Match_result.png", img_match)
    
    print("Process completed. Check your results.")
    