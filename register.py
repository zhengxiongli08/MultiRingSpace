
# Author: Zhengxiong Li

# Multiple scale ring space algorithm

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import cv2 as cv
import pickle
import numpy as np
import argparse
import shutil
from logger import Logger
from preprocess import read_slide, monomer_preprocess, polysome_preprocess
from conv import get_mask_list, get_conv_list, get_diff_list
from keypoint import get_kps, get_color_keypoint_img
from eigen import get_eigens
from match import Matching


# Functions
def get_params():
    """
    Receive parameters from terminal
    """
    parser = argparse.ArgumentParser(description="Indicate parameters, use --help for help.")
    parser.add_argument("--group_path", type=str, default="../BiopsyDatabase/WSI_100Cases/BC-1-group1", help="group's path")
    parser.add_argument("--result_path", type=str, default="../result", help="result's folder")
    parser.add_argument("--slide_type", type=str, default="monomer", help="slide type, monomer/polysome")
    parser.add_argument("--radius_min", type=int, default=5, help="minimum radius of ring")
    parser.add_argument("--radius_max", type=int, default=30, help="maximum radius of ring")
    parser.add_argument("--thickness", type=int, default=7, help="thickness of the ring")
    parser.add_argument("--resize_height", type=int, default=1024, help="image's height after resize")
    # Initialize parser
    args = parser.parse_args()
    # Extract database path
    database_path = os.path.dirname(args.group_path)
    # Extract group name
    group_name = os.path.basename(args.group_path)
    # Extract slide information
    slides_list = list()
    for file_name in os.listdir(args.group_path):
        if file_name.endswith("svs"):
            slides_list.append(file_name)
    if len(slides_list) != 2:
        raise Exception(f"{len(slides_list)} slides detected, which should be 2")
    slide_1_path = os.path.join(args.group_path, slides_list[0])
    slide_2_path = os.path.join(args.group_path, slides_list[1])
    # Extract slide landmarks information
    landmarks_path = os.path.join(args.group_path, "landmarks")
    landmarks_folders = os.listdir(landmarks_path)
    if landmarks_folders[0] in slides_list[0] and landmarks_folders[1] in slides_list[1]:
        landmarks_1_path = os.path.join(landmarks_path, landmarks_folders[0])
        landmarks_2_path = os.path.join(landmarks_path, landmarks_folders[1])
    elif landmarks_folders[0] in slides_list[1] and landmarks_folders[1] in slides_list[0]:
        landmarks_1_path = os.path.join(landmarks_path, landmarks_folders[1])
        landmarks_2_path = os.path.join(landmarks_path, landmarks_folders[0])
    else:
        raise Exception("Landmarks not found.")
    # Extract magnification
    if "magnification" in group_name:
        magnification = "40x"
    else:
        magnification = "20x"
    # Clean up result folder
    if os.path.exists(args.result_path):
        shutil.rmtree(args.result_path)
    os.mkdir(args.result_path)
    # Put them into a dictionary
    params = dict()
    params["database_path"] = database_path
    params["group_path"] = args.group_path
    params["group_name"] = group_name
    params["result_path"] = args.result_path
    params["slide_type"] = args.slide_type
    params["radius_min"] = args.radius_min
    params["radius_max"] = args.radius_max
    params["thickness"] = args.thickness
    params["resize_height"] = args.resize_height
    params["slide_1_path"] = slide_1_path
    params["slide_2_path"] = slide_2_path
    params["slide_1_name"] = slides_list[0]
    params["slide_2_name"] = slides_list[1]
    params["landmarks_1_path"] = landmarks_1_path
    params["landmarks_2_path"] = landmarks_2_path
    params["magnification"] = magnification
    
    return params

def compute(params, myLogger):
    # Get necessary information
    slide_num = params["slide_num"]
    if (slide_num == 1):
        slide_path = params["slide_1_path"]
    elif (slide_num == 2):
        slide_path = params["slide_2_path"]
    else:
        raise Exception("Invalid slide_num")

    slide_type = params["slide_type"]
    radius_min = params["radius_min"]
    radius_max = params["radius_max"]
    thickness = params["thickness"]
    result_path = params["result_path"]
    resize_height = params["resize_height"]
    
    # Begin to compute keypoints and eigenvectors
    myLogger.print(f"Start processing for slide {slide_num}.")
    # Read image and preprocess it
    img_origin = read_slide(slide_path, resize_height)
    myLogger.print(f"Read slide {params[f'slide_{slide_num}_name']} done.")
    if (slide_type == "monomer"):
        img_origin_gray, img_nobg, img_nobg_gray = monomer_preprocess(img_origin)
    elif (slide_type == "polysome"):
        img_origin_gray, img_nobg, img_nobg_gray = polysome_preprocess(img_origin)
    myLogger.print("Preprocess completed successfully!")
    myLogger.print(f"Image's size: {img_nobg_gray.shape}")
            
    # Get list of convolution kernels
    mask_list = get_mask_list(radius_min, radius_max, thickness)
    myLogger.print("Get mask list successfully!")
    
    # Get convolution pyramid
    conv_list = get_conv_list(img_nobg_gray, mask_list)
    myLogger.print("Get convolution list successfully!")
    
    # Get differential pyramid
    diff_list = get_diff_list(conv_list)
    myLogger.print("Get differential list successfully!")
    
    # Get keypoints
    kps = get_kps(diff_list)
    myLogger.print("Get keypoints successfully!")
    myLogger.print(f"Total keypoints number for slide {slide_num}: {kps.shape[0]}")
    
    # Get eigenvectors
    eigens = get_eigens(img_origin_gray, kps, mask_list)
    myLogger.print("Get eigen vectors successfully!")
    myLogger.print(f"Eigen vectors' shape for slide {slide_num}: {eigens.shape}")
    
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
        temp = cv.applyColorMap(diff_list[i], cv.COLORMAP_WINTER)
        cv.imwrite(diff_img_path, temp)
    # Save keypoints map
    img_color = get_color_keypoint_img(img_nobg, kps)
    kp_color_path = os.path.join(slide_result_path, f"img_kp_color.png")
    cv.imwrite(kp_color_path, img_color)
    
    return kps, eigens

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
    myLogger.print(f"Start image registration using Multiple Ring Space algorithm.")
    myLogger.print("{:<20} {}".format("Parameter", "Value"))
    for i in params:
        myLogger.print("{:<20} {}".format(i, params[i]))
    myLogger.print()
    
    # Start processing
    # Process for slide 1 & 2
    params_1 = params.copy()
    params_2 = params.copy()
    params_1["slide_num"] = 1
    params_2["slide_num"] = 2
    kps_1, eigens_1 = compute(params_1, myLogger)
    kps_2, eigens_2 = compute(params_2, myLogger)
    
    # Match them
    match_kps_1, match_kps_2 = Matching(kps_1, eigens_1, kps_2, eigens_2)
    
    # Save data for evaluation
    eva_data_path = os.path.join(result_path, "eva_data")
    os.makedirs(eva_data_path, exist_ok=True)
    # Save match keypoints
    match_kps_1_path = os.path.join(eva_data_path, "match_kps_1.npy")
    match_kps_2_path = os.path.join(eva_data_path, "match_kps_2.npy")
    np.save(match_kps_1_path, match_kps_1)
    np.save(match_kps_2_path, match_kps_2)
    # Save parameters dictionary
    params_path = os.path.join(eva_data_path, "params.pkl")
    with open(params_path, "wb") as params_file:
        pickle.dump(params, params_file)

    myLogger.print(f"Process complete. Check your results in {result_path}.")
    
    return

if __name__ == "__main__":
    register()
    
    print("Program finished!")
