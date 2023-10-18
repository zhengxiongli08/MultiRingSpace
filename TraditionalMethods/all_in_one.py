# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

# Match images by a traditional algorithm. 

import os
import cv2 as cv
import numpy as np
import argparse
from logger import Logger
from preprocess import monomer_preprocess, polysome_preprocess


# Main function
def get_params(params: dict):
    """
    Get parameters from terminal.
    """
    parser = argparse.ArgumentParser(description="Indicate the slide group's path and slide type(monomer/polysome).")
    parser.add_argument("--group_path", type=str, default="../../BiopsyDatabase/monomer/case6-group3", help="slide group's path")
    parser.add_argument("--result_path", type=str, default="../../result", help="results' folder")
    parser.add_argument("--slide_type", type=str, default="monomer", help="slide type, monomer/polysome")
    parser.add_argument("--method", type=str, default="akaze", help="akaze/kaze/brisk/orb/sift")
    parser.add_argument("--weight", type=float, default=0.8, help="weight used for figure out good match")
    # Put them in a dictionary
    args = parser.parse_args()
    params["group_path"] = args.group_path
    params["result_path"] = args.result_path
    params["slide_type"] = args.slide_type
    params["method"] = args.method
    params["weight"] = args.weight
    group_path = params["group_path"]
    # Get the exact path of 2 slides
    slide_list = list()
    for file in os.listdir(group_path):
        if file.endswith(".mrxs"):
            slide_list.append(file)
            
    slide_1_path = os.path.join(group_path, slide_list[0])
    slide_2_path = os.path.join(group_path, slide_list[1])
    params["slide_1_path"] = slide_1_path
    params["slide_2_path"] = slide_2_path

    return


def main():
    # Read parameters from terminal
    params = {}
    get_params(params)
    # Get necessary information
    path_1 = params["slide_1_path"]
    path_2 = params["slide_2_path"]
    result_path = params["result_path"]
    slide_type = params["slide_type"]
    method = params["method"]
    weight = params["weight"]
    myLogger = Logger(result_path)
    # Record dict's information
    myLogger.print(f"Start image registration using {method} algorithm.")
    myLogger.print("{:<20} {}".format("Parameter", "Value"))
    for i in params:
        myLogger.print("{:<20} {}".format(i, params[i]))
    myLogger.print()
    
    # Preprocess images
    match slide_type:
        case "monomer":
            img_1_origin, img_1_nobg, img_1_nobg_gray = monomer_preprocess(path_1)
            img_2_origin, img_2_nobg, img_2_nobg_gray = monomer_preprocess(path_2)
        case "polysome":
            img_1_origin, img_1_nobg, img_1_nobg_gray = polysome_preprocess(path_1)
            img_2_origin, img_2_nobg, img_2_nobg_gray = polysome_preprocess(path_2)

    # Match these images
    match method:
        case "akaze":
            detector = cv.AKAZE_create()
        case "kaze":
            detector = cv.KAZE_create()
        case "brisk":
            detector = cv.BRISK_create()
        case "orb":
            detector = cv.ORB_create()
        case "sift":
            detector = cv.SIFT_create()
            
    bf = cv.BFMatcher()
    [kp_1, des_1] = detector.detectAndCompute(img_1_nobg_gray, None)
    [kp_2, des_2] = detector.detectAndCompute(img_2_nobg_gray, None)

    img_1_kp = np.zeros_like(img_1_nobg)
    img_2_kp = np.zeros_like(img_2_nobg)
    cv.drawKeypoints(img_1_nobg, kp_1, img_1_kp, color = (0, 255, 0))
    cv.drawKeypoints(img_2_nobg, kp_2, img_2_kp, color = (0, 255, 0))

    matches = bf.knnMatch(des_1, des_2, k = 2)

    # select some good matches
    good_match = []
    for m,n in matches:
        if m.distance < weight*n.distance:
            good_match.append([m])
            
    img_match = cv.drawMatchesKnn(img_1_nobg, kp_1, img_2_nobg, kp_2, good_match, None, flags=2)
    
    # Save the intermediate and final results
    img_1_origin_path = os.path.join(result_path, "img_1_origin.png")
    img_2_origin_path = os.path.join(result_path, "img_2_origin.png")
    img_1_nobackground_path = os.path.join(result_path, "img_1_nobackground.png")
    img_2_nobackground_path = os.path.join(result_path, "img_2_nobackground.png")
    img_1_kp_path = os.path.join(result_path, "img_1_kp.png")
    img_2_kp_path = os.path.join(result_path, "img_2_kp.png")
    img_match_path = os.path.join(result_path, "img_match.png")
    
    cv.imwrite(img_1_origin_path, img_1_origin)
    cv.imwrite(img_2_origin_path, img_2_origin)
    cv.imwrite(img_1_nobackground_path, img_1_nobg)
    cv.imwrite(img_2_nobackground_path, img_2_nobg)
    cv.imwrite(img_1_kp_path, img_1_kp)
    cv.imwrite(img_2_kp_path, img_2_kp)
    cv.imwrite(img_match_path, img_match)
    myLogger.print(f"{method} registration finished!")

    return
    
if __name__ == "__main__":
    main()
    