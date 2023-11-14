
# Author: Zhengxiong Li

# Multiple scale ring space algorithm

import argparse
import os
import taichi as ti
import cv2 as cv
import pickle
import numpy as np
from logger import Logger
from preprocess import read_slide, monomer_preprocess, polysome_preprocess
from conv import get_mask_list, get_conv_list, get_diff_list
from keypoint import get_kps, get_color_keypoint_img
from eigen import get_eigens
from match import Matching


PARAMS = dict()

# Functions
def get_params():
    """
    Receive parameters from terminal
    """
    parser = argparse.ArgumentParser(description="Indicate parameters, use --help for help.")
    parser.add_argument("--groups_path", type=str, default="../BiopsyDatabase", help="slide groups' path")
    parser.add_argument("--group_path", type=str, default="../BiopsyDatabase/monomer/case6-group4", help="slide group's path")
    parser.add_argument("--result_path", type=str, default="../result", help="result's folder")
    parser.add_argument("--slide_type", type=str, default="monomer", help="slide type, monomer/polysome")
    parser.add_argument("--radius_min", type=int, default=5, help="minimum radius of ring")
    parser.add_argument("--radius_max", type=int, default=30, help="maximum radius of ring")
    parser.add_argument("--thickness", type=int, default=7, help="thickness of the ring")
    parser.add_argument("--dim_lower_bound", type=int, default=2000, help="image's height lower bound")
    parser.add_argument("--dim_upper_bound", type=int, default=8000, help="image's height upper bound")
    parser.add_argument("--resize_height", type=int, default=1024, help="image's height after resize")
    # Put them into a dictionary
    args = parser.parse_args()
    # PARAMS["groups_path"] = args.groups_path
    # PARAMS["group_path"] = args.group_path
    PARAMS["groups_path"] = "../BiopsyDatabase/WSI_100Cases"
    PARAMS["group_path"] = "../BiopsyDatabase/WSI_100Cases/TM-2-40magnification-group3"
    PARAMS["result_path"] = args.result_path
    PARAMS["slide_type"] = args.slide_type
    PARAMS["radius_min"] = args.radius_min
    PARAMS["radius_max"] = args.radius_max
    PARAMS["thickness"] = args.thickness
    PARAMS["dim_lower_bound"] = args.dim_lower_bound
    PARAMS["dim_upper_bound"] = args.dim_upper_bound
    PARAMS["resize_height"] = args.resize_height
    # Get the exact path of 2 slides
    group_path = PARAMS["group_path"]
    slides_list = list()
    for file in os.listdir(group_path):
        if file.endswith(".mrxs") or file.endswith(".svs"):
            slides_list.append(file)
    
    slide_1_path = os.path.join(group_path, slides_list[0])
    slide_2_path = os.path.join(group_path, slides_list[1])
    PARAMS["slide_1_path"] = slide_1_path
    PARAMS["slide_2_path"] = slide_2_path
    
    return
    
def compute(num: int, myLogger: Logger):
    """
    Get keypoints and eigenvector
    """
    # Get necessary information
    if (num == 1):
        path = PARAMS["slide_1_path"]
    elif (num == 2):
        path = PARAMS["slide_2_path"]
        
    result_path = PARAMS["result_path"]
    slide_type = PARAMS["slide_type"]
    radius_min = PARAMS["radius_min"]
    radius_max = PARAMS["radius_max"]
    thickness = PARAMS["thickness"]
    
    # Begin compute keypoints and eigenvector
    myLogger.print(f"Start processing for slide {num}.")
    # Read image and preprocess it
    img_origin = read_slide(path, PARAMS)
    if (slide_type == "monomer"):
        img_nobg, img_nobg_gray = monomer_preprocess(img_origin)
    elif (slide_type == "polysome"):
        img_nobg, img_nobg_gray = polysome_preprocess(img_origin)
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
    
    # Get eigenvectors
    eigens = get_eigens(img_nobg_gray, kps, mask_list)
    myLogger.print("Get eigen vectors successfully!")
    
    # Save results
    slide_result_path = os.path.join(result_path, f"slide-{num}")
    if not os.path.exists(slide_result_path):
        os.mkdir(slide_result_path)
    # Save original, no background, nobg & gray scale image
    img_origin_path = os.path.join(slide_result_path, "img_origin.png")
    img_nobg_path = os.path.join(slide_result_path, "img_nobg.png")
    img_nobg_gray_path = os.path.join(slide_result_path, "img_nobg_gray.png")
    cv.imwrite(img_origin_path, img_origin)
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
    myLogger.print(f"Total keypoints number for slide {num}: {kps.shape[0]}")
    img_color = get_color_keypoint_img(img_nobg, kps)
    kp_color_path = os.path.join(slide_result_path, f"img_kp_color.png")
    cv.imwrite(kp_color_path, img_color)
    
    return kps, eigens

def main():
    """
    Match 2 slides using multiple scale ring space algorithm
    """
    # Read parameters from terminal
    get_params()
    # Record dict's information
    result_path = PARAMS["result_path"]
    os.makedirs(result_path, exist_ok=True)
    myLogger = Logger(result_path)
    myLogger.print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    myLogger.print(f"Start image registration using Multiple Ring Space algorithm.")
    myLogger.print("{:<20} {}".format("Parameter", "Value"))
    for i in PARAMS:
        myLogger.print("{:<20} {}".format(i, PARAMS[i]))
    myLogger.print()
    
    # Start processing
    # Process for slide 1 & 2
    kps_1, eigens_1 = compute(1, myLogger)
    kps_2, eigens_2 = compute(2, myLogger)
    
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
        pickle.dump(PARAMS, params_file)

    myLogger.print(f"Process complete. Check your results in {result_path}.")
    
    return

if __name__ == "__main__":
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    ti.init(arch=ti.gpu)
    main()
        