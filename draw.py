
# This program is designed to draw the match image

import cv2 as cv
import numpy as np
import pickle


# Functions
def draw_line(img_1, img_2, kps_1, kps_2):
    """
    Draw lines between corresponding keypoints
    Order of coordinates is (coor_h, coor_w)
    But the order of cv.line is (coor_w, coor_h)
    """
    # Line's color
    # GREEN = (0, 255, 0)
    
    # Determine shift value for coor_w
    shift = img_1.shape[1]
    # Combine the 2 images
    img = np.hstack([img_1, img_2])
    # Draw lines
    for i in range(0, kps_1.shape[0]):
        color = np.random.randint(0, 256, 3, dtype=np.uint8).tolist()
        kp_1_coor_h, kp_1_coor_w = kps_1[i, 0], kps_1[i, 1]
        kp_2_coor_h, kp_2_coor_w = kps_2[i, 0], kps_2[i, 1]
        coor_left = (kp_1_coor_w, kp_1_coor_h)
        coor_right = (kp_2_coor_w + shift, kp_2_coor_h)
        img = cv.line(img, coor_left, coor_right, color=color, thickness=1)
    
    return img

def rotate(img_1, img_2, kps_1, kps_2):
    """
    Rotate the second image to the same direction of the first image
    """
    H, Mask = cv.estimateAffinePartial2D(kps_2, kps_1, cv.RANSAC)
    print(H)
    print(Mask)
    
    return

if __name__ == "__main__":
    img_1 = cv.imread("./temp/img_nobg_1.png")
    img_2 = cv.imread("./temp/img_nobg_2.png")
    with open("./temp/result_a.pkl", "rb") as file:
        kps_1 = pickle.load(file)
    with open("./temp/result_b.pkl", "rb") as file:
        kps_2 = pickle.load(file)
    
    # result = draw_line(img_1, img_2, kps_1, kps_2)
    # cv.imwrite("./temp/result.png", result)
    
    rotate(img_1, img_2, kps_1, kps_2)
