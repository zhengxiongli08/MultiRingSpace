
# Automatically run all data group

import os
import shutil
import subprocess
from datetime import datetime


# Global constants
MONOMER_PATH = "../BiopsyDatabase/monomer"
POLYSOME_PATH = "../BiopsyDatabase/polysome"
GOLDCASE_PATH = "../BiopsyDatabase/WSI_100Cases"
RESULT_PATH = "../result"
ARCHIVE_PATH = "../archive"

# Functions
def my_run(command: str):
    """
    A warp for subprocess.check_call function
    """
    command_new = command.split()
    subprocess.check_call(command_new)

    return

def main():
    """
    Automation script
    """
    # Get current time
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    # Get path for archive
    archive_folder_path = os.path.join(ARCHIVE_PATH, current_time)
    # Get path for monomer results and polysome results
    monomer_result_path = os.path.join(archive_folder_path, "monomer")
    polysome_result_path = os.path.join(archive_folder_path, "polysome")
    goldcase_result_path  = os.path.join(archive_folder_path, "goldcase")
    matches_result_path = os.path.join(archive_folder_path, "matches")
    os.makedirs(matches_result_path, exist_ok=True)
    
    
    # Do it for monomer dataset
    for group_name in os.listdir(MONOMER_PATH):
        group_path = os.path.join(MONOMER_PATH, group_name)
        # Remove the result folder and create a new one
        if os.path.exists(RESULT_PATH):
            shutil.rmtree(RESULT_PATH)
        os.mkdir(RESULT_PATH)
        my_run(f"python main.py \
                --group_path={group_path} \
                --result_path={RESULT_PATH} \
                --slide_type=monomer")
        # Get path for case's result
        group_result_path = os.path.join(monomer_result_path, group_name+"-result")
        # Copy results from result folder to archive folder
        shutil.copytree(RESULT_PATH, group_result_path)
        # Save match image into a single folder
        img_match_path = os.path.join(RESULT_PATH, "img_match.png")
        target_path = os.path.join(matches_result_path, f"{group_name}.png")
        shutil.copy(img_match_path, target_path)

    
    # Do it for polysome dataset
    for group_name in os.listdir(POLYSOME_PATH):
        group_path = os.path.join(POLYSOME_PATH, group_name)
        # Remove the result folder and create a new one
        if os.path.exists(RESULT_PATH):
            shutil.rmtree(RESULT_PATH)
        os.mkdir(RESULT_PATH)
        my_run(f"python main.py \
                --group_path={group_path} \
                --result_path={RESULT_PATH} \
                --slide_type=polysome")
        # Get path for case's result
        group_result_path = os.path.join(polysome_result_path, group_name+"-result")
        # Copy results from result folder to archive folder
        shutil.copytree(RESULT_PATH, group_result_path)
        # Save match image into a single folder
        img_match_path = os.path.join(RESULT_PATH, "img_match.png")
        target_path = os.path.join(matches_result_path, f"{group_name}.png")
        shutil.copy(img_match_path, target_path)
        
    # Do it for gold case dataset
    for group_name in os.listdir(GOLDCASE_PATH):
        group_path = os.path.join(GOLDCASE_PATH, group_name)
        # Remove the result folder and create a new one
        if os.path.exists(RESULT_PATH):
            shutil.rmtree(RESULT_PATH)
        os.mkdir(RESULT_PATH)
        my_run(f"python main.py \
                --group_path={group_path} \
                --result_path={RESULT_PATH} \
                --slide_type=monomer")
        # Get path for case's result
        group_result_path = os.path.join(goldcase_result_path, group_name+"-result")
        # Copy results from result folder to archive folder
        shutil.copytree(RESULT_PATH, group_result_path)
        # Save match image into a single folder
        img_match_path = os.path.join(RESULT_PATH, "img_match.png")
        target_path = os.path.join(matches_result_path, f"{group_name}.png")
        shutil.copy(img_match_path, target_path)
        
    return
    
if __name__ == "__main__":
    main()
