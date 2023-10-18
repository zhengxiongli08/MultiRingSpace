
# This program is used to remove background for monomer slides and polysome slides

import openslide
import rembg
import numpy as np
import cv2 as cv
from PIL import Image


def convert(img):
    """
    Convert image format from RGB into BGR
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

def img_preprocess(img):
    """
    Remove background from the original image.
    """
    img_new = rembg.remove(img)
    
    return img_new

def preprocess(path):
    """
    Read slide, and preprocess it.
    Including remove background and transform it into a gray scale image.
    """
    img_origin = read_slide(path)
    img_nobg = img_preprocess(img_origin)
    img_nobg_gray = cv.cvtColor(img_nobg, cv.COLOR_BGR2GRAY)
    
    return img_origin, img_nobg, img_nobg_gray
    

if __name__ == "__main__":
    img = preprocess("../BiopsyDatabase/monomer/case1-group1/slide-2022-12-19T17-02-13-R5-S1.mrxs")
    cv.imwrite("./result.png", img)
    