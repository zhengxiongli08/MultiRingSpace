
# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

# Multiple scale ring space

import argparse
import os
import taichi as ti
import cv2 as cv
import pickle
from logger import Logger
from preprocess import monomer_preprocess, polysome_preprocess
from conv import get_mask_list, get_conv_list, get_diff_list
from keypoint import get_kps, get_color_keypoint_img
from eigen import get_eigens
from match import Matching


# Functions
def get_params(params: dict):
    """
    Receive parameters from terminal
    """
    parser = argparse.ArgumentParser(description="Indicate parameters, use --help for help.")
    parser.add_argument("--group_path", type=str, default="../BiopsyDatabase/monomer/case6-group3", help="slide group's path")
    parser.add_argument("--result_path", type=str, default="../result", help="result's folder")
    parser.add_argument("--slide_type", type=str, default="monomer", help="slide type, monomer/polysome")
    parser.add_argument("--radius_min", type=int, default=5, help="minimum radius of ring")
    parser.add_argument("--radius_max", type=int, default=30, help="maximum radius of ring")
    parser.add_argument("--thickness", type=int, default=7, help="thickness of the ring")
    parser.add_argument("--weight", type=float, default=0.8, help="weight used for figure out good match")
    # Put them into a dictionary
    args = parser.parse_args()
    params["group_path"] = args.group_path
    params["result_path"] = args.result_path
    params["slide_type"] = args.slide_type
    params["radius_min"] = args.radius_min
    params["radius_max"] = args.radius_max
    params["thickness"] = args.thickness
    params["weight"] = args.weight
    # Get the exact path of 2 slides
    group_path = params["group_path"]
    slide_list = list()
    for file in os.listdir(group_path):
        if file.endswith(".mrxs"):
            slide_list.append(file)
    
    slide_1_path = os.path.join(group_path, slide_list[0])
    slide_2_path = os.path.join(group_path, slide_list[1])
    params["slide_1_path"] = slide_1_path
    params["slide_2_path"] = slide_2_path
    
    return
    
def compute(params: dict, num: int, myLogger: Logger):
    """
    Get keypoints and eigenvector
    """
    # Get necessary information
    match num:
        case 1: 
            path = params["slide_1_path"]
        case 2:
            path = params["slide_2_path"]
    result_path = params["result_path"]
    slide_type = params["slide_type"]
    radius_min = params["radius_min"]
    radius_max = params["radius_max"]
    thickness = params["thickness"]
    
    # Begin compute keypoints and eigenvector
    myLogger.print(f"Start processing for slide {num}.")
    # Read image and preprocess it
    if (slide_type == "monomer"):
        img_origin, img_nobg, img_nobg_gray = monomer_preprocess(path)
    elif (slide_type == "polysome"):
        img_origin, img_nobg, img_nobg_gray = polysome_preprocess(path)
    myLogger.print("Preprocess completed successfully!")
            
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
    
    # Get eigenvector
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
    myLogger.print(f"Total keypoints number: {kps.shape[0]}")
    img_color = get_color_keypoint_img(img_nobg, kps)
    kp_color_path = os.path.join(slide_result_path, f"img_kp_color.png")
    cv.imwrite(kp_color_path, img_color)
    
    
    return kps, eigens
    
#     good_matches = match.bf_match(eigens_1, eigens_2, threshold=0.7)
    
#     # 画出结果
#     img_match = cv.drawMatchesKnn(img_1, keypoints_1, img_2, keypoints_2, good_matches, None, flags=2)
    
#     cv.imwrite(folder_path + "/Match_result.png", img_match)
    
#     print("Process completed. Check your results.")

def main():
    """
    Match 2 slides using multiple ring space algorithm
    """
    # Read parameters from terminal
    params = {}
    get_params(params)
    # Record dict's information
    result_path = params["result_path"]
    myLogger = Logger(result_path)
    myLogger.print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    myLogger.print(f"Start image registration using Multiple Ring Space algorithm.")
    myLogger.print("{:<20} {}".format("Parameter", "Value"))
    for i in params:
        myLogger.print("{:<20} {}".format(i, params[i]))
    myLogger.print()
    
    # Process for slide 1 & 2
    kps_1, eigens_1 = compute(params, 1, myLogger)
    kps_2, eigens_2 = compute(params, 2, myLogger)
    
    # Match them
    MatchResultA, MatchResultB = Matching(kps_1, eigens_1, kps_2, eigens_2)
    
    return

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    ti.init(arch=ti.gpu)
    main()
        