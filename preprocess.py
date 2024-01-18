
# This program is used for preprocess

import rembg
import cv2 as cv
import numpy as np
import os
from skimage import io
from numba import njit, prange


def my_resize(img, new_h, method=cv.INTER_AREA):
    """
    Scale the image proportionally
    """
    height, width = img.shape[:2]
    new_w = int((width / height) * new_h)
    result = cv.resize(img, (new_w, new_h), interpolation=method)  # Attention: method can be changed

    return result
    
def read_slide(slide_path):
    """
    Open slide using skimage
    """
    slide = io.imread(slide_path)
    # Convert the color
    img = cv.cvtColor(slide, cv.COLOR_RGB2BGR)

    return img

def bg_remove(img_origin):
    """
    Remove the background using rembg
    Input and output images are BGR channels
    """
    # Convert color
    img_rgb = cv.cvtColor(img_origin, cv.COLOR_BGR2RGB)
    # Get the mask for it
    temp = rembg.remove(img_rgb, only_mask=True)
    mask = np.where(temp > 100, 1, 0)
    # Remove background
    img_nobg = np.multiply(img_rgb, mask[:, :, np.newaxis]).astype(np.uint8)
    # Convery color
    img_nobg = cv.cvtColor(img_nobg, cv.COLOR_RGB2BGR)
    
    return img_nobg, mask

def trans_gray(img, stain_type):
    """
    Transform the BGR image into gray scale
    Local Histogram Equalization is used based on the stain type of this slide
    """
    if stain_type == "HE":
        clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    elif stain_type == "IHC":
        clahe = cv.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    else:
        raise Exception("Stain type not supported.")
    # Get the gray scale image
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img_gray = clahe.apply(img_gray)
    
    return img_gray

@njit(parallel=True)
def bg_remove2(img):
    """
    Remove background pixels based on the order of BGR values
    Accelerated using numba
    """
    img_nobg = img.copy()
    # Remove background
    height, width = img.shape[:2]
    for i in prange(0, height):
        for j in prange(0, width):
            b = int(img[i, j][0])
            g = int(img[i, j][1])
            r = int(img[i, j][2])
            yellow_flag = (r > g) and (g > b)
            blue_flag = (b > g) and (g > r)
            purple_flag = (b > g) and (r > g)
            
            if (yellow_flag or blue_flag or purple_flag):
                pass
            else:
                img_nobg[i, j] = [0, 0, 0]

    return img_nobg

def polysome_preprocess(img_origin):
    """
    Read a polysome slide, and preprocess it.
    Including remove background and transform it into a gray scale image.
    """
    # Get gray scale for image_origin
    img_origin_gray = cv.cvtColor(img_origin, cv.COLOR_BGR2GRAY)
    # Remove background
    img_nobg = bg_remove(img_origin)
    # Transform it into gray scale
    img_nobg_gray = cv.cvtColor(img_nobg, cv.COLOR_BGR2GRAY)
                
    return img_origin_gray, img_nobg, img_nobg_gray

def find_slide_path(group_path, file_type=".svs"):
    """
    Determine the slide paths based on the group path
    If there is a HE slide in the group, it should be the first
    """
    # Find the slides
    slide_paths = list()
    for file in sorted(os.listdir(group_path)):
        if file.endswith(file_type):
            slide_path = os.path.join(group_path, file)
            slide_paths.append(slide_path)
    # Check slides number
    if len(slide_paths) != 2:
        raise Exception(f"{len(slide_paths)} slides detected, which should be 2")
    # Handle the HE slide in advance, if it exists
    if "HE" in slide_paths[1]:
        result = (slide_paths[1], slide_paths[0])
    else:
        result = (slide_paths[0], slide_paths[1])

    return result

def find_landmarks_path(slide_path):
    """
    Determine the landmarks path based on the slide path
    """
    # Determine the group path
    group_path = os.path.dirname(slide_path)
    # Determine the landmarks path
    landmarks_path = os.path.join(group_path, "landmarks")
    landmarks_folders = os.listdir(landmarks_path)
    # Determine the slide name
    slide_name = os.path.splitext(os.path.basename(slide_path))[0]
    # Get result
    if slide_name in landmarks_folders[0]:
        result = os.path.join(landmarks_path, landmarks_folders[0])
    elif slide_name in landmarks_folders[1]:
        result = os.path.join(landmarks_path, landmarks_folders[1])
    else:
        raise Exception("Landmarks not found.")
    
    return result

def find_magnification(group_path):
    """
    Determine the magnification of this group
    """
    if "magnification" in group_path:
        magnification = "40x"
    else:
        magnification = "20x"
    
    return magnification

def find_stain_type(slide_path):
    """
    Determine the stain type of this slide
    """
    slide_name = os.path.basename(slide_path)
    if "HE" in slide_name:
        result = "HE"
    else:
        result = "IHC"

    return result
