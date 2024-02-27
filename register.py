
# Author: Zhengxiong Li

# Multiple scale ring space algorithm

import sys
import os
import cv2 as cv
import numpy as np
import argparse
import shutil
import json
from natsort import natsorted
from concurrent.futures import ProcessPoolExecutor
from logger import Logger
from preprocess import *
from conv import get_mask_list, get_conv_list, get_diff_list
from keypoint import get_kps, get_kps_from_mask, get_color_keypoint_img
from eigen import get_eigens
from match import Matching_TwoMapping


# Functions
def get_params():
    """
    Receive parameters from terminal
    """
    long_str1 = "/mnt/Disk1/whole_slide_image_analysis/Lizhengxiong/Projects/MultiRingSpace/BiopsyDatabase/WSI_100Cases/BC-9-group2"
    long_str2 = "/mnt/Disk1/whole_slide_image_analysis/Lizhengxiong/Projects/MultiRingSpace/result"
    parser = argparse.ArgumentParser(description="Indicate parameters, use --help for help.")
    parser.add_argument("--group_path", type=str, default=long_str1, help="group's path")
    parser.add_argument("--result_path", type=str, default=long_str2, help="result's folder")
    parser.add_argument("--conv_radius_min", type=int, default=2, help="minimum radius of ring for convolution")
    parser.add_argument("--conv_radius_max", type=int, default=6, help="maximum radius of ring for convolution")
    parser.add_argument("--eigen_radius_min_large", type=int, default=100, help="minimum radius of ring for descriptors of large image")
    parser.add_argument("--eigen_radius_max_large", type=int, default=180, help="maximum radius of ring for descriptors of large image")
    parser.add_argument("--eigen_radius_min_small", type=int, default=6, help="minimum radius of ring for descriptors of small image")
    parser.add_argument("--eigen_radius_max_small", type=int, default=12, help="maximum radius of ring for descriptors of small image")
    parser.add_argument("--thickness", type=int, default=10, help="thickness of the ring")
    parser.add_argument("--overlap_factor", type=int, default=3, help="overlap factor for multiple rings")
    parser.add_argument("--resize_height_large", type=int, default=1024, help="large image's height for slide 1")
    parser.add_argument("--resize_height_small", type=int, default=128, help="small image's height for slide 1")
    parser.add_argument("--keypoint_radius", type=int, default=1, help="radius of keypoints detect region")
    # Initialize parser
    args = parser.parse_args()
    # Clean up result folder
    if os.path.exists(args.result_path):
        shutil.rmtree(args.result_path)
    os.mkdir(args.result_path)
    # Put them into a dictionary
    params = dict()
    params["group_path"] = args.group_path
    params["result_path"] = args.result_path
    params["conv_radius_min"] = args.conv_radius_min
    params["conv_radius_max"] = args.conv_radius_max
    params["eigen_radius_min_large"] = args.eigen_radius_min_large
    params["eigen_radius_max_large"] = args.eigen_radius_max_large
    params["eigen_radius_min_small"] = args.eigen_radius_min_small
    params["eigen_radius_max_small"] = args.eigen_radius_max_small
    params["thickness"] = args.thickness
    params["overlap_factor"] = args.overlap_factor
    params["resize_height_large"] = args.resize_height_large
    params["resize_height_small"] = args.resize_height_small
    params["keypoint_radius"] = args.keypoint_radius
    
    return params

def compute(img_origin, 
            img_origin_gray, 
            img_nobg, 
            img_nobg_gray, 
            params, 
            myLogger, 
            slide_num, 
            mask=None,
            diff_list_large=None):
    # Get necessary information
    conv_radius_min = params["conv_radius_min"]
    conv_radius_max = params["conv_radius_max"]
    if (slide_num == 3) or (slide_num == 4):
        eigen_radius_min = params["eigen_radius_min_small"]
        eigen_radius_max = params["eigen_radius_max_small"]
    else:
        eigen_radius_min = params["eigen_radius_min_large"]
        eigen_radius_max = params["eigen_radius_max_large"]
    thickness = params["thickness"]
    overlap_factor = params["overlap_factor"]
    result_path = params["result_path"]
    keypoint_radius = params["keypoint_radius"]
    
    # Begin to compute keypoints and eigen vectors
    myLogger.print(f"Start processing for slide {slide_num}.")
    myLogger.print(f"Image's size: {img_nobg_gray.shape}")
    
    # Get list of convolution kernels
    conv_mask_list = get_mask_list(conv_radius_min, conv_radius_max)
    myLogger.print("Get mask list for convolution done.")
    
    # Get convolution pyramid
    conv_list = get_conv_list(img_nobg_gray, conv_mask_list)
    myLogger.print("Get convolution list done.")
    
    # Get differential pyramid
    if (slide_num == 3) or (slide_num == 4):
        diff_list = list()
        for diff_img in diff_list_large:
            temp = my_resize(diff_img, img_nobg_gray.shape[0])
            diff_list.append(temp)
    else:
        diff_list = get_diff_list(conv_list, img_nobg_gray)
    myLogger.print("Get differential list done.")
    
    # Get keypoints
    if (slide_num == 3) or (slide_num == 4):
        myLogger.print("Using mask to generate keypoints.")
        kps = get_kps_from_mask(mask)
    else:
        kps = get_kps(diff_list, keypoint_radius)
        myLogger.print("Get keypoints done.")
    myLogger.print(f"Total keypoints number for slide: {kps.shape[0]}")
    
    # Get eigenvectors
    img_mean = np.zeros_like(img_nobg_gray).astype(np.int32)
    # Get image mean
    for diff_img in diff_list:
        img_mean += diff_img
    img_mean = img_mean / len(diff_list)
    img_mean = img_mean.astype(np.uint8)
    # Get conv kernels for eigens
    if (slide_num == 3) or (slide_num == 4):
        eigen_mask_list = get_mask_list(eigen_radius_min, eigen_radius_max, 1, overlap_factor)
        eigens_list = list()
        for diff_img in diff_list:
            temp = get_eigens(diff_img, kps, eigen_mask_list)
            eigens_list.append(temp)
        eigens = np.hstack(eigens_list)
    else:
        eigen_mask_list = get_mask_list(eigen_radius_min, eigen_radius_max, thickness, overlap_factor)
        eigens = get_eigens(img_mean, kps, eigen_mask_list)
    myLogger.print("Get eigen vectors done.")
    myLogger.print(f"Eigen vectors' shape: {eigens.shape}")
    
    # Save results
    slide_result_path = os.path.join(result_path, f"slide-{slide_num}")
    os.makedirs(slide_result_path, exist_ok=True)
    # Save original, no background, nobg & gray scale image
    img_origin_path = os.path.join(slide_result_path, "img_origin.png")
    img_origin_gray_path = os.path.join(slide_result_path, "img_origin_gray.png")
    img_nobg_path = os.path.join(slide_result_path, "img_nobg.png")
    img_nobg_gray_path = os.path.join(slide_result_path, "img_nobg_gray.png")
    cv.imwrite(img_origin_path, img_origin)
    cv.imwrite(img_origin_gray_path, img_origin_gray)
    cv.imwrite(img_nobg_path, img_nobg)
    cv.imwrite(img_nobg_gray_path, img_nobg_gray)
    # Save convolution images
    for i in range(0, len(conv_list)):
        conv_img_path = os.path.join(slide_result_path, f"img_conv_{i}.png")
        cv.imwrite(conv_img_path, conv_list[i])
    # Save differntial images
    for i in range(0, len(diff_list)):
        diff_img_path = os.path.join(slide_result_path, f"img_diff_{i}.png")
        cv.imwrite(diff_img_path, diff_list[i])
    # Save keypoints map
    img_color = get_color_keypoint_img(img_nobg, kps)
    kp_color_path = os.path.join(slide_result_path, f"img_kp_color.png")
    cv.imwrite(kp_color_path, img_color)
    
    # Temp save
    img_mean_path = os.path.join(slide_result_path, f"img_mean.png")
    cv.imwrite(img_mean_path, img_mean)
    
    return kps, eigens, diff_list

def register():
    """
    Match 2 slides using multiple scale ring space algorithm
    """
    # Record dict's information
    params = get_params()
    print(f"Start processing {params['group_path']}")
    result_path = params["result_path"]
    myLogger = Logger(result_path)
    myLogger.print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    myLogger.print(f"Start image registration using Multiple Scale Ring Space Algorithm.")
    myLogger.print("{:<30} {}".format("Parameter", "Value"))
    for i in params:
        myLogger.print("{:<30} {}".format(i, params[i]))
    myLogger.print()
    
    # Start processing
    # Get necessary parameters
    slide_1_path, slide_2_path = find_slide_path(params['group_path'])
    stain_type_1 = find_stain_type(slide_1_path)
    stain_type_2 = find_stain_type(slide_2_path)
    resize_h_l = params["resize_height_large"]
    resize_h_s = params["resize_height_small"]
    # Read both slides and resize them
    slide_1 = read_slide(slide_1_path)
    img_origin_1_large = my_resize(slide_1, resize_h_l)
    slide_1_h = slide_1.shape[0]
    del slide_1
    slide_2 = read_slide(slide_2_path)
    resize_h_l_2 = int(slide_2.shape[0] * (resize_h_l / slide_1_h))
    resize_h_s_2 = int(resize_h_l_2 * (resize_h_s / resize_h_l))
    img_origin_2_large = my_resize(slide_2, resize_h_l_2)
    del slide_2
    # Preprocess images
    img_nobg_1_large, mask_1_large = bg_remove(img_origin_1_large)
    img_nobg_2_large, mask_2_large = bg_remove(img_origin_2_large)
    img_origin_gray_1_large = trans_gray(img_origin_1_large, stain_type_1)
    img_origin_gray_2_large = trans_gray(img_origin_2_large, stain_type_2)
    img_nobg_gray_1_large = trans_gray(img_nobg_1_large, stain_type_1)
    img_nobg_gray_2_large = trans_gray(img_nobg_2_large, stain_type_2)

    img_origin_1_small = my_resize(img_origin_1_large, resize_h_s)
    img_origin_gray_1_small = my_resize(img_origin_gray_1_large, resize_h_s)
    img_nobg_1_small = my_resize(img_nobg_1_large, resize_h_s)
    img_nobg_gray_1_small = my_resize(img_nobg_gray_1_large, resize_h_s)
    mask_1_small = my_resize(mask_1_large, resize_h_s, cv.INTER_NEAREST)

    img_origin_2_small = my_resize(img_origin_2_large, resize_h_s_2)
    img_origin_gray_2_small = my_resize(img_origin_gray_2_large, resize_h_s_2)
    img_nobg_2_small = my_resize(img_nobg_2_large, resize_h_s_2)
    img_nobg_gray_2_small = my_resize(img_nobg_gray_2_large, resize_h_s_2)
    mask_2_small = my_resize(mask_2_large, resize_h_s_2, cv.INTER_NEAREST)
    
    # Process for slide 1 & 2
    kps_1_large, eigens_1_large, diff_list_1 = compute(img_origin_1_large, 
                                          img_origin_gray_1_large, 
                                          img_nobg_1_large, 
                                          img_nobg_gray_1_large, 
                                          params, 
                                          myLogger, 
                                          1)
    
    kps_2_large, eigens_2_large, diff_list_2 = compute(img_origin_2_large, 
                                          img_origin_gray_2_large, 
                                          img_nobg_2_large, 
                                          img_nobg_gray_2_large, 
                                          params, 
                                          myLogger, 
                                          2)
    
    kps_1_small, eigens_1_small, _ = compute(img_origin_1_small, 
                                          img_origin_gray_1_small, 
                                          img_nobg_1_small, 
                                          img_nobg_gray_1_small, 
                                          params, 
                                          myLogger, 
                                          3, 
                                          mask_1_small,
                                          diff_list_1)
    
    kps_2_small, eigens_2_small, _ = compute(img_origin_2_small, 
                                          img_origin_gray_2_small, 
                                          img_nobg_2_small, 
                                          img_nobg_gray_2_small, 
                                          params, 
                                          myLogger, 
                                          4, 
                                          mask_2_small,
                                          diff_list_2)
    
    # Match them
    myLogger.print("Matching...")
    match_kps_11, match_kps_22, _, _, match_kps_1, match_kps_2 = Matching_TwoMapping(kps_1_small, 
                                                               eigens_1_small, 
                                                               kps_2_small, 
                                                               eigens_2_small, 
                                                               (resize_h_l / resize_h_s), 
                                                               kps_1_large, 
                                                               eigens_1_large, 
                                                               kps_2_large, 
                                                               eigens_2_large)
    
    # Save data for evaluation
    eva_data_path = os.path.join(result_path, "eva_data")
    os.makedirs(eva_data_path, exist_ok=True)
    # Save match keypoints
    match_kps_1_path = os.path.join(eva_data_path, "match_kps_1.npy")
    match_kps_2_path = os.path.join(eva_data_path, "match_kps_2.npy")
    match_kps_11_path = os.path.join(eva_data_path, "match_kps_11.npy")
    match_kps_22_path = os.path.join(eva_data_path, "match_kps_22.npy")
    np.save(match_kps_1_path, match_kps_1)
    np.save(match_kps_2_path, match_kps_2)
    np.save(match_kps_11_path, match_kps_11)
    np.save(match_kps_22_path, match_kps_22)
    # Save parameters dictionary
    params_json = os.path.join(eva_data_path, "params.json")
    with open(params_json, "w") as file:
        json.dump(params, file)
    myLogger.print(f"Process complete. Check your results in {result_path}.")
    
    return

if __name__ == "__main__":
    register()
    
    print("Program finished!")
