
# Automatically run gold cases in parallel

import os
import shutil
import subprocess
import concurrent.futures
from datetime import datetime
from evaluate import evaluate
from natsort import natsorted
from logger import Logger
from register import register


# Global constants
GOLDCASE_PATH = "../BiopsyDatabase/WSI_100Cases"
EVALUATE_READY_PATH = "../archive/evaluate_ready"

# Functions
def main():
    # Dicts for input parameters
    params_list = list()
    for group_name in natsorted(os.listdir(GOLDCASE_PATH)):
        params = dict()
        group_path = os.path.join(GOLDCASE_PATH, group_name)
        slides_list = list()
        for file_name in os.listdir(group_path):
            if file_name.endswith(".svs"):
                slides_list.append(file_name)
        # Get path of slides
        slide_1_path = os.path.join(group_path, slides_list[0])
        slide_2_path = os.path.join(group_path, slides_list[1])
        # Get path for result to evaluate
        result_path = os.path.join(EVALUATE_READY_PATH, f"{group_name}-result")
        if os.path.exists(result_path):
            shutil.rmtree(result_path)
        os.makedirs(result_path)
        # Add information to dict
        params["groups_path"] = GOLDCASE_PATH
        params["group_path"] = group_path
        params["group_name"] = group_name
        params["result_path"] = result_path
        params["slide_1_name"] = slides_list[0]
        params["slide_2_name"] = slides_list[1]
        params["slide_1_path"] = slide_1_path
        params["slide_2_path"] = slide_2_path
        params["slide_num"] = 0
        params["slide_type"] = "monomer"
        params["radius_min"] = 5
        params["radius_max"] = 30
        params["thickness"] = 7
        params["resize_height"] = 1024
        # Add dict into list
        params_list.append(params)
        
    # Execute registration in parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        executor.map(register, params_list)

if __name__ == "__main__":
    main()

    print("Program finished!")
