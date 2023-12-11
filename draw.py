
# This program is designed to draw the match image

import cv2 as cv
import numpy as np


# Functions
def draw_line(img_1, img_2, kps_1, kps_2, thickness=1):
    """
    Draw lines between corresponding keypoints
    Order of coordinates is (coor_h, coor_w)
    But the order of cv.line is (coor_w, coor_h)
    Both images should have the same height
    """
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
        img = cv.line(img, coor_left, coor_right, color=color, thickness=thickness)
    
    return img

if __name__ == "__main__":
    
    print("Program finished!")
