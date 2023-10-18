# This program is used to remove background for monomer slides and polysome slides

import openslide
import rembg
import numpy as np
import cv2 as cv
from PIL import Image


def convert(img):
    """
    Function:
        Convert image format from RGB into BGR
    
    Since openslide read slides using PIL package, 
    image's channels order is "RGB". 
    But opencv image's channel order is "BGR". 
    """
    img_new = np.zeros_like(img)
    img_new[:, :, 0] = img[:, :, 2]
    img_new[:, :, 1] = img[:, :, 1]
    img_new[:, :, 2] = img[:, :, 0]
    
    return img_new

def read_slide(path, dimension=5):
    """
    Open slide using package "openslide"
    """
    slide = openslide.open_slide(path)
    resolution = slide.level_dimensions[dimension]
    img = slide.get_thumbnail(resolution)
    img = np.array(img)
    img = convert(img)
    
    return img

def monomer_preprocess(path):
    """
    Read a monomer slide, and preprocess it.
    Including remove background and transform it into a gray scale image.
    """
    img_origin = read_slide(path)
    temp = rembg.remove(img_origin)
    # After processing using rembg, there are 4 channels, we only need 3 (BGR)
    channels = cv.split(temp)
    img_nobg = cv.merge(channels[:3])
    img_nobg_gray = cv.cvtColor(temp, cv.COLOR_BGR2GRAY)
    
    return img_origin, img_nobg, img_nobg_gray

def polysome_preprocess(path):
    """
    Read a polysome slide, and preprocess it.
    Including remove background and transform it into a gray scale image.
    """
    img_origin = read_slide(path)
    img_nobg = img_origin.copy()
    # Remove background
    height = img_origin.shape[0]
    width = img_origin.shape[1]
    for i in range(0, height):
        for j in range(0, width):
            b = int(img_origin[i, j][0])
            g = int(img_origin[i, j][1])
            r = int(img_origin[i, j][2])
            yellow_flag = (r > g) and (g > b)
            blue_flag = (b > g) and (g > r)
            purple_flag = (b > g) and (r > g)
            
            if (yellow_flag or blue_flag or purple_flag):
                pass
            else:
                img_nobg[i, j] = [0, 0, 0]
    # Transform it into gray scale
    img_nobg_gray = cv.cvtColor(img_nobg, cv.COLOR_BGR2GRAY)
                
    return img_origin, img_nobg, img_nobg_gray
