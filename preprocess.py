# This program is used to remove background for monomer slides and polysome slides

import openslide
import rembg
import numpy as np
import cv2 as cv
import numba
import pyvips
from PIL import Image
from numba import njit, prange


def convert(img):
    """
    Function:
        Convert image format from RGB into BGR in place
    
    Since openslide read slides using PIL package, 
    image's channels order is "RGB". 
    But opencv image's channel order is "BGR". 
    """
    img[:, :, 2], img[:, :, 0] = img[:, :, 0], img[:, :, 2]
    
    return 

def read_slide(slide_path, resize_height):
    """
    Open slide using pyvips
    """
    img = pyvips.Image.new_from_file(slide_path, level=0)
    img = np.asarray(img)[:, :, :3]
    convert(img)
    # Resize the image
    height, width = img.shape[:2]
    new_height = resize_height
    new_width = int((width / height) * new_height)
    result = cv.resize(img, (new_width, new_height))
    
    return result

def monomer_preprocess(img_origin):
    """
    Read a monomer slide, and preprocess it.
    Including remove background and transform it into a gray scale image.
    """
    # Get gray scale for image_origin
    img_origin_gray = cv.cvtColor(img_origin, cv.COLOR_BGR2GRAY)
    temp = rembg.remove(img_origin)
    # After processing using rembg, there are 4 channels, we only need 3 (BGR)
    channels = cv.split(temp)
    img_nobg = cv.merge(channels[:3])
    img_nobg_gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    
    return img_origin_gray, img_nobg, img_nobg_gray

@njit(parallel=True)
def bg_remove(img):
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

if __name__ == "__main__":
    print("Program finished.")
    