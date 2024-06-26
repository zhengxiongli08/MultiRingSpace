
# This program is used to evaluate the results of matching.

import json
import os
import numpy as np
import cv2 as cv
import pyvips
from logger import Logger
from natsort import natsorted
from draw import draw_line, my_hstack
from preprocess import *
from lxx import transform_match_location


# Functions
def read_json(json_path):
    """
    Read a single .json file
    Attention: The coordinates in json file's order is different with the normal one
    flip it
    """
    # Open the json file
    with open(json_path, 'r') as file:
        json_data = json.load(file)
    # Extract the useful data
    landmarks_data = json_data["Models"]["LandMarkListModel"]["Points"][0]["LabelList"]
    # Sort the data by label number
    landmarks_data = natsorted(landmarks_data, key=lambda x: x["Label"])
    # Convert dict into a numpy array
    coordinates = [item["Position"] for item in landmarks_data]
    coordinates = np.array(coordinates)[:, :-1]
    # Flip 2 columns
    coordinates = np.fliplr(coordinates)

    return coordinates

def read_jsons(jsons_path):
    """
    Read multiple .jsons files and calculate the mean value of coordinates
    """
    # Record the number of json files and positions data
    count = 0
    coordinates_list = list()
    for json_file in os.listdir(jsons_path):
        # If it's not a json file, skip it
        if (not json_file.endswith(".json")):
            continue
        # If it's a json file, read it
        json_path = os.path.join(jsons_path, json_file)
        coordinates = read_json(json_path)
        coordinates_list.append(coordinates)
        count += 1
    # Calculate the mean value of coordinates
    sum_coordinates = np.zeros_like(coordinates_list[0])
    for item in coordinates_list:
        sum_coordinates += item
    mean_coordinates = sum_coordinates / count
    
    return mean_coordinates

def get_resize_factor(slide_path, img_height):
    """
    Calculate the resize factor
    size_new = size_old * resize_factor
    """
    slide = pyvips.Image.new_from_file(slide_path, level=0)
    slide_height = slide.height
    resize_factor = img_height / slide_height
    
    return resize_factor

def get_coords(jsons_path, resize_factor):
    """
    Get manual coordinates of keypoints given by humans
    Scale it using the resize factor
    """
    # Get mean coordinates from json fils
    mean_coordinates = read_jsons(jsons_path)
    # Calculate final coordinates
    final_coordinates = mean_coordinates * resize_factor
    final_coordinates = final_coordinates.astype(np.int32)
    
    return final_coordinates

def affine_transform(kps_1, kps_2, img_2):
    """
    Execute affine transformation on a given image
    kps_1 and kps_2 are used to calculate the affine matrix
    img_2 is a floating image
    No resize process
    """
    # Estimate the affine matrix, from kps_2(floating) to kps_1(fixed)
    affine_matrix, _ = cv.estimateAffine2D(kps_2, kps_1, cv.RANSAC)
    # Get shape of image
    height, width = img_2.shape[:2]
    # Affine transformation
    img_2_new = cv.warpAffine(src=img_2, M=affine_matrix, dsize=(width, height))
    # Calculate new keypoints coordinates
    kps_2_new = cv.transform(kps_2.reshape((-1, 1, 2)), affine_matrix)
    kps_2_new = np.squeeze(kps_2_new, axis=1)
    
    return img_2_new, kps_2_new, affine_matrix

def pixel2um(pixel_error, magnification):
    """
    Transform pixel error into um error
    """
    if magnification == "20x":
        um_error = pixel_error * 0.5
    elif magnification == "40x":
        um_error = pixel_error * 0.25
    else:
        raise Exception("Unsupported magnification factor.")
    
    return um_error

def quantize(coords_1, coords_2, affine_matrix, resize_factor, magnification):
    """
    Quantize the results of image registration
    We transform img2(floating) to img1(fixed),
    So it should be resize_factor_1
    """
    # Do affine transformation for manual coordinates
    coords_2_new = cv.transform(coords_2.reshape((-1, 1, 2)), affine_matrix)
    coords_2_new = np.squeeze(coords_2_new, axis=1)
    # Calculate Euclidean Distances and pixel errors
    pixel_errors = np.linalg.norm(coords_1 - coords_2_new, axis=1)
    top_pixel_errors = pixel_errors / resize_factor
    # Calculate the real world errors in um
    um_errors = np.zeros_like(pixel_errors)
    for i in range(0, um_errors.shape[0]):
        um_errors[i] = pixel2um(top_pixel_errors[i], magnification)
    
    return pixel_errors, um_errors

def quantize2(coords_1, coords_2, kps_1, kps_2, resize_factor, magnification):
    coords_2_new, _ = transform_match_location(coords_2, kps_2, kps_1, 500)
    # Calculate Euclidean Distances and pixel errors
    pixel_errors = np.linalg.norm(coords_1 - coords_2_new, axis=1)
    top_pixel_errors = pixel_errors / resize_factor
    # Calculate the real world errors in um
    um_errors = np.zeros_like(pixel_errors)
    for i in range(0, um_errors.shape[0]):
        um_errors[i] = pixel2um(top_pixel_errors[i], magnification)
    
    return pixel_errors, um_errors

def random_rows(kps_1, kps_2, rows_num=100):
    """
    Randomly extract some rows from current matrix
    """
    total_rows = kps_1.shape[0]
    if rows_num > total_rows:
        return kps_1, kps_2
    # Generate random indices to select rows
    random_indices = np.random.choice(total_rows, size=rows_num, replace=False)
    # Extract rows
    res_1 = kps_1[random_indices, :]
    res_2 = kps_2[random_indices, :]

    return res_1, res_2

def evaluate(result_path):
    """
    Evaluate the performance of image registration
    """
    # Read parameters dict
    dict_path = os.path.join(result_path, "eva_data", "params.json")
    with open(dict_path, "r") as params_file:
        params = json.load(params_file)
    # Extract necessary parameters
    group_path = params["group_path"]
    slide_1_path, slide_2_path = find_slide_path(group_path)
    landmarks_path_1 = find_landmarks_path(slide_1_path)
    landmarks_path_2 = find_landmarks_path(slide_2_path)
    magnification = find_magnification(group_path)
    eva_data_path = os.path.join(result_path, "eva_data")
    kps_1_path = os.path.join(eva_data_path, "match_kps_1.npy")
    kps_2_path = os.path.join(eva_data_path, "match_kps_2.npy")
    img_1_path = os.path.join(result_path, "slide-1", "img_origin.png")
    img_2_path = os.path.join(result_path, "slide-2", "img_origin.png")
    # Read keypoints data from npy files
    kps_1 = np.load(kps_1_path).astype(np.int32)
    kps_2 = np.load(kps_2_path).astype(np.int32)
    # Read thumbnails
    img_1 = cv.imread(img_1_path)
    img_2 = cv.imread(img_2_path)
    # Get the manual landmarks
    resize_factor_1 = get_resize_factor(slide_1_path, img_1.shape[0])
    resize_factor_2 = get_resize_factor(slide_2_path, img_2.shape[0])
    coords_1 = get_coords(landmarks_path_1, resize_factor_1)
    coords_2 = get_coords(landmarks_path_2, resize_factor_2)
    # Create logger
    myLogger = Logger(result_path)
    
    # Draw lines before affine transformation
    img_combo = my_hstack(img_1, img_2)
    kps_1_draw, kps_2_draw = random_rows(kps_1, kps_2, 100)
    img_match = draw_line(img_1, img_2, kps_1_draw, kps_2_draw, thickness=2)
    img_match_manual = draw_line(img_1, img_2, coords_1, coords_2, thickness=3)
    img_match_combo = np.vstack((img_match_manual, img_match))
    # Do affine transformation
    _, _, affine_matrix = affine_transform(kps_1, kps_2, img_2)
    
    # Quantize
    pixel_errors, um_errors = quantize(coords_1, coords_2, affine_matrix, resize_factor_1, magnification)
    pixel_errors2, um_errors2 = quantize2(coords_1, coords_2, kps_1, kps_2, resize_factor_1, magnification)
    
    pixel_error = np.mean(pixel_errors)
    um_error = np.mean(um_errors)
    
    myLogger.print("Evaluation globally:")
    myLogger.print(f"Max pixel error: {np.max(pixel_errors):.2f} pixels.")
    myLogger.print(f"Min pixel error: {np.min(pixel_errors):.2f} pixels.")
    myLogger.print(f"Max um error: {np.max(um_errors):.2f}um.")
    myLogger.print(f"Min um error: {np.min(um_errors):.2f}um.")
    myLogger.print(f"Pixel error: {pixel_error:.2f} pixels.")
    myLogger.print(f"um error: {um_error:.2f}um")
    
    myLogger.print("Evaluation locally:")
    myLogger.print(f"Max pixel error: {np.max(pixel_errors2):.2f} pixels.")
    myLogger.print(f"Min pixel error: {np.min(pixel_errors2):.2f} pixels.")
    myLogger.print(f"Max um error: {np.max(um_errors2):.2f}um.")
    myLogger.print(f"Min um error: {np.min(um_errors2):.2f}um.")
    myLogger.print(f"Pixel error: {np.mean(pixel_errors2):.2f} pixels.")
    myLogger.print(f"um error: {np.mean(um_errors2):.2f}um")
    
    # Save results
    eva_result_path = os.path.join(result_path, "eva_result")
    os.makedirs(eva_result_path, exist_ok=True)
    # Results before affine transformation
    img_combo_path = os.path.join(eva_result_path, "img_combo.png")
    img_match_path = os.path.join(eva_result_path, "img_match.png")
    img_match_manual_path = os.path.join(eva_result_path, "img_match_manual.png")
    img_match_combo_path = os.path.join(eva_result_path, "img_match_combo.png")
    cv.imwrite(img_combo_path, img_combo)
    cv.imwrite(img_match_path, img_match)
    cv.imwrite(img_match_manual_path, img_match_manual)
    cv.imwrite(img_match_combo_path, img_match_combo)
    # Results of errors
    errors = dict()
    errors["status"] = "successful"
    errors["pixel_errors"] = list(pixel_errors)
    errors["um_errors"] = list(um_errors)
    errors["pixel_errors_lxx"] = list(pixel_errors2)
    errors["um_errors_lxx"] = list(um_errors2)
    errors_path = os.path.join(eva_result_path, "errors.json")
    with open(errors_path, "w") as json_file:
        json.dump(errors, json_file, indent=4)

    return
    
if __name__ == '__main__':
    evaluate("../result")
    
    print("Program finished!")
