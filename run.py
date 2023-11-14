
# Automatically run gold cases and evaluate the results

import os
import shutil
import subprocess
from datetime import datetime


# Global constants
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
    goldcase_result_path  = os.path.join(archive_folder_path, "goldcase")
    
    # Do it for gold case dataset
    for group_name in os.listdir(GOLDCASE_PATH):
        group_path = os.path.join(GOLDCASE_PATH, group_name)
        # Remove the result folder and create a new one
        if os.path.exists(RESULT_PATH):
            shutil.rmtree(RESULT_PATH)
        os.mkdir(RESULT_PATH)
        my_run(f"python main.py \
                --groups_path={GOLDCASE_PATH} \
                --group_path={group_path} \
                --result_path={RESULT_PATH} \
                --slide_type=monomer")
        # Get path for case's result
        group_result_path = os.path.join(goldcase_result_path, group_name+"-result")
        # Copy results from result folder to archive folder
        shutil.copytree(RESULT_PATH, group_result_path)
        
    return
    
if __name__ == "__main__":
    main()
