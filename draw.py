
# This program is designed to draw the match image

import cv2 as cv
import numpy as np
import pickle
import numba
from numba import njit, prange
from math import pi


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

@njit
def get_line_angle(kp_1, kp_2):
    """
    Calcualte the angle of line and axis X (height direction)
    Axis X is height direction
    Axis Y is width direction
    """
    x_1, y_1 = kp_1
    x_2, y_2 = kp_2
    if (x_1 == x_2):
        if (y_2 > y_1):
            return 0.5*pi
        else:
            return 1.5*pi
    temp = (y_2 - y_1) / (x_2 - x_1)
    angle = np.arctan(temp)
    # Deal with corner case
    if (y_2 > y_1):
        if (x_2 < x_1):
            angle += pi
    elif (y_2 < y_1):
        if (x_2 < x_1):
            angle += pi
        else:
            angle += 2 * pi
    else:
        if (x_2 < x_1):
            angle = pi
    
    return angle

@njit(parallel=True)
def get_rotate_angle(kps_1, kps_2):
    """
    Calculate the rotation angle.
    Methodology:
    1. Take 2 points from kps_1
    2. Take 2 corresponding points from kps_2
    3. Calculate line angle for kps_1
    4. Calculate line angle for kps_2
    5. Calculate the difference between them
    6. Repeat it until all combos are used
    7. Calculate the mean value as the rotation angle
    """
    # Get match keypoints length
    kp_length = kps_1.shape[0]
    angles_length = int(kp_length * (kp_length - 1) / 2)
    angles = np.zeros(angles_length)
    for i in prange(0, kp_length):
        for j in prange(1, kp_length-i):
            angle_1 = get_line_angle(kps_1[i], kps_1[i+j])
            angle_2 = get_line_angle(kps_2[i], kps_2[i+j])
            angle = angle_2 - angle_1
            index = (2 * kp_length - i - 1) * i / 2 + j - 1
            index = int(index)
            angles[index] = angle
    print(f"Total length: {angles_length}")
    rotate_angle = np.mean(angles)
    
    return rotate_angle

# def rotate(img_1, img_2, kps_1, kps_2):
#     """
#     Rotate the second image to the same direction of the first image
#     """
#     H, Mask = cv.estimateAffinePartial2D(kps_2, kps_1, cv.RANSAC)
#     height, width = img_2.shape[:2]
#     new_img = cv.warpAffine(src=img_2, M=H, dsize=(width, height))
#     print(H)
#     cv.imwrite("./temp/trans.png", new_img)
    
#     return

def img_rotate(src, angle):
    """逆时针旋转图像任意角度
    Rotate input image in counterclockwise direction
    Input angle should be radian value
    """
    h,w = src.shape[:2]
    center = (w//2, h//2)
    angle_degree = angle / pi * 180
    rotate_mat = cv.getRotationMatrix2D(center, angle_degree, 1.0)
    # 调整旋转后的图像长宽
    rotated_h = int((w * np.abs(rotate_mat[0,1]) + (h * np.abs(rotate_mat[0,0]))))
    rotated_w = int((h * np.abs(rotate_mat[0,1]) + (w * np.abs(rotate_mat[0,0]))))
    rotate_mat[0,2] += (rotated_w - w) // 2
    rotate_mat[1,2] += (rotated_h - h) // 2
    # 旋转图像
    rotated_img = cv.warpAffine(src, rotate_mat, (rotated_w,rotated_h))

    return rotated_img

if __name__ == "__main__":
    img_1 = cv.imread("./temp/img_nobg_1.png")
    img_2 = cv.imread("./temp/img_nobg_2.png")
    with open("./temp/result_a.pkl", "rb") as file:
        kps_1 = pickle.load(file)
    with open("./temp/result_b.pkl", "rb") as file:
        kps_2 = pickle.load(file)
    
    # kps_1 = np.random.uniform(-10, 10, (20000, 2))
    # kps_2 = np.random.uniform(-10, 10, (20000, 2))
    
    rotate_angle = get_rotate_angle(kps_1, kps_2)
    # print(rotate_angle)
    
    
    # img = cv.imread("./temp/result.png")
    rotated_img = img_rotate(img_2, -rotate_angle)
    cv.imwrite("./temp/img_2_rotated.png", rotated_img)
