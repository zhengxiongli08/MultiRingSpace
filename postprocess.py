
# After registration and evaluation, postprocess the results

import os
import json
import numpy as np
from natsort import natsorted


# Global constants
ARCHIVE_PATH = "../archive/warehouse"

# Functions
def postprocess(exp_path):
    pixel_errors = list()
    um_errors = list()
    for group in natsorted(os.listdir(exp_path)):
        group_path = os.path.join(exp_path, group)
        if not os.path.isdir(group_path):
            continue
        errors_path = os.path.join(group_path, "eva_result", "errors.json")
        with open(errors_path, "r") as json_file:
            errors = json.load(json_file)
        pixel_errors.append(errors["pixel_error"])
        um_errors.append(errors["um_error"])
    
    pixel_error_mean = np.mean(pixel_errors)
    um_error_mean = np.mean(um_errors)
    post_res_path = os.path.join(exp_path, "post_result.json")
    with open(post_res_path, "w") as res_file:
        json.dump({"pixel_error_mean": pixel_error_mean, "um_error_mean": um_error_mean}, res_file)
    
    return pixel_error_mean, um_error_mean

if __name__ == "__main__":
    # Get the latest experiment results' path
    temp = natsorted(os.listdir(ARCHIVE_PATH), reverse=True)[0]
    exp_path = os.path.join(ARCHIVE_PATH, temp)
    # Postprocess
    pixel_error_mean, um_error_mean = postprocess(exp_path)
    print(f"Mean pixel error: {pixel_error_mean:.2f} pixels.")
    print(f"Mean um error: {um_error_mean:.2f}um")
    
    print("Program finished.")
