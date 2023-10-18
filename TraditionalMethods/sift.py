# Author: Zhengxiong Li
# Email: zhengxiong_li@foxmail.com

# Match images by SIFT algorithm. 

import os
import cv2 as cv
import numpy as np
from preprocess import preprocess
import argparse
from logger import Logger


# Global variables
PARAMS = {}
myLogger = Logger()
RESULT_PATH = "../result"


# Main function
def get_params():
    """
    Get parameters from terminal.
    """
    parser = argparse.ArgumentParser(description="Indicate the slide group's path and slide type(monomer/polysome).")
    parser.add_argument("--group_path", type=str, default="../BiopsyDatabase/monomer/case1-group1", help="slide group's path")
    parser.add_argument("--slide_type", type=str, default="monomer", help="slide type, monomer/polysome")
    parser.add_argument("--weight", type=float, default=0.75, help="weight used for figure out good match")
    # Put them in a dictionary
    args = parser.parse_args()
    PARAMS["group_path"] = args.group_path
    PARAMS["slide_type"] = args.slide_type
    PARAMS["weight"] = args.weight
    group_path = PARAMS["group_path"]
    # Get the exact path of 2 slides
    slide_list = list()
    for file in os.listdir(group_path):
        if file.endswith(".mrxs"):
            slide_list.append(file)
            
    slide_1_path = os.path.join(group_path, slide_list[0])
    slide_2_path = os.path.join(group_path, slide_list[1])
    PARAMS["slide_1_path"] = slide_1_path
    PARAMS["slide_2_path"] = slide_2_path
    # Record dict's information
    myLogger.print("Start image registration using SIFT algorithm.")
    myLogger.print("{:<20} {}".format("Parameter", "Value"))
    for i in PARAMS:
        myLogger.print("{:<20} {}".format(i, PARAMS[i]))
    myLogger.print()

    return


def main():
    # Read parameters from terminal
    get_params()
    path_1 = PARAMS["slide_1_path"]
    path_2 = PARAMS["slide_2_path"]
    weight = PARAMS["weight"]
    
    # Preprocess images
    img_1_origin, img_1_nobg, img_1_nobg_gray = preprocess(path_1)
    img_2_origin, img_2_nobg, img_2_nobg_gray = preprocess(path_2)

    # Match these images
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
    img_1_origin_path = os.path.join(RESULT_PATH, "img_1_origin.png")
    img_2_origin_path = os.path.join(RESULT_PATH, "img_2_origin.png")
    img_1_nobackground_path = os.path.join(RESULT_PATH, "img_1_nobackground.png")
    img_2_nobackground_path = os.path.join(RESULT_PATH, "img_2_nobackground.png")
    img_1_kp_path = os.path.join(RESULT_PATH, "img_1_kp.png")
    img_2_kp_path = os.path.join(RESULT_PATH, "img_2_kp.png")
    img_match_path = os.path.join(RESULT_PATH, "img_match.png")
    
    cv.imwrite(img_1_origin_path, img_1_origin)
    cv.imwrite(img_2_origin_path, img_2_origin)
    cv.imwrite(img_1_nobackground_path, img_1_nobg)
    cv.imwrite(img_2_nobackground_path, img_2_nobg)
    cv.imwrite(img_1_kp_path, img_1_kp)
    cv.imwrite(img_2_kp_path, img_2_kp)
    cv.imwrite(img_match_path, img_match)
    myLogger.print("SIFT registration finished!")

    return
    
if __name__ == "__main__":
    main()
    