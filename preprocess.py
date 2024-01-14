
# This program is used to remove background for monomer slides and polysome slides

import rembg
import cv2 as cv
import numpy as np
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

def monomer_preprocess(img_origin, stain_type):
    """
    Read a monomer slide, and preprocess it.
    Including remove background and transform it into a gray scale image.
    Input and outpur channels follow opencv rules (BGR)
    """
    # Convert color
    img_rgb = cv.cvtColor(img_origin, cv.COLOR_BGR2RGB)
    # Get the mask for it
    temp = rembg.remove(img_rgb, only_mask=True)
    mask = np.where(temp > 100, 1, 0)
    # Remove background
    img_nobg = np.multiply(img_rgb, mask[:, :, np.newaxis]).astype(np.uint8)
    # # Remove extra channels
    # channels = cv.split(img_nobg)
    # img_nobg = cv.cvtColor(cv.merge(channels[:3]), cv.COLOR_RGB2BGR)
    # Transform it into gray scale based on the stain type
    if stain_type == "HE":
        clahe = cv.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
    elif stain_type == "IHC":
        clahe = cv.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
    else:
        raise Exception("Stain type not supported.")
    # img_nobg_gray = cv.cvtColor(img_nobg, cv.COLOR_BGR2GRAY)
    img_nobg_gray = cv.cvtColor(img_nobg, cv.COLOR_RGB2GRAY)
    img_nobg_gray = clahe.apply(img_nobg_gray)
    # Get gray scale for image_origin
    img_origin_gray = cv.cvtColor(img_origin, cv.COLOR_BGR2GRAY)
    img_origin_gray = clahe.apply(img_origin_gray)

    return img_origin_gray, img_nobg, img_nobg_gray, mask

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
    import os
    test_path = "../BiopsyDatabase/WSI_100Cases/BC-23-40magnification-group1"
    for file in os.listdir(test_path):
        if not file.endswith(".svs"):
            continue
        slide_path = os.path.join(test_path, file)
        img = read_slide(slide_path, 1024)
        print(img.shape)

    print("Program finished.")
    