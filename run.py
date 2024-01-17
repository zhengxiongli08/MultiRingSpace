
# Automatically run gold cases in parallel

import os
import shutil
import subprocess
import multiprocessing
from datetime import datetime
from natsort import natsorted
from postprocess import postprocess


# Global constants
GOLDCASE_PATH = "../BiopsyDatabase/WSI_100Cases"
ARCHIVE_PATH = "../archive/warehouse"
EVALUATE_READY_PATH = "../archive/eva_ready"

# Functions
def my_run(command):
    subprocess.run(command, shell=True)

    return

def main():
    # Clean up the eva_ready directory
    if os.path.exists(EVALUATE_READY_PATH):
        shutil.rmtree(EVALUATE_READY_PATH)
    os.mkdir(EVALUATE_READY_PATH)
    # Get path for long-term storage
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
    des_path = os.path.join(ARCHIVE_PATH, formatted_time)
    # Commands for registration    
    commands_reg = list()
    commands_eva = list()
    
    # Scan database
    for group_name in natsorted(os.listdir(GOLDCASE_PATH)):
        group_path = os.path.join(GOLDCASE_PATH, group_name)
        result_path = os.path.join(EVALUATE_READY_PATH, group_name)
        command_reg = (
            f"python register.py "
            f"--group_path={group_path} "
            f"--result_path={result_path}"
        )
        command_eva = (
            f"python evaluate.py "
            f"--result_path={result_path}"
        )
        
        commands_reg.append(command_reg)
        commands_eva.append(command_eva)
    
    # Execute registration
    with multiprocessing.Pool(processes=4) as pool:
        pool.map_async(my_run, commands_reg)
        pool.close()
        pool.join()
    
    # Execute evaluation
    with multiprocessing.Pool(processes=12) as pool:
        pool.map_async(my_run, commands_eva)
        pool.close()
        pool.join()
    
    shutil.copytree(EVALUATE_READY_PATH, des_path)
    
    # Postprocess
    pixel_error_mean, um_error_mean = postprocess(des_path)
    print(f"Mean pixel error: {pixel_error_mean:.2f} pixels.")
    print(f"Mean um error: {um_error_mean:.2f}um")
    
    return

if __name__ == "__main__":
    main()

    print("Program finished!")
