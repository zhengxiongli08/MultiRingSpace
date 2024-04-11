
# Automatically run gold cases in parallel

import os
import shutil
import subprocess
import multiprocessing
import argparse
from datetime import datetime
from natsort import natsorted
from postprocess import postprocess
from evaluate import evaluate


# Global constants
GOLDCASE_PATH = "../BiopsyDatabase/WSI_100Cases"
ARCHIVE_PATH = "../archive/warehouse"
EVALUATE_READY_PATH = "../archive/eva_ready"

# Functions
def my_run(command):
    subprocess.run(command, shell=True)

    return

def mode_register():
    """
    Mode for registration
    """
    # Clean up the eva_ready directory
    if os.path.exists(EVALUATE_READY_PATH):
        shutil.rmtree(EVALUATE_READY_PATH)
    os.mkdir(EVALUATE_READY_PATH)
    
    # Commands for registration    
    jobs = list()
    result_paths = list()
    
    # Scan database
    for group_name in natsorted(os.listdir(GOLDCASE_PATH)):
        group_path = os.path.join(GOLDCASE_PATH, group_name)
        result_path = os.path.join(EVALUATE_READY_PATH, group_name)
        result_paths.append(result_path)
        command_reg = (
            f"python register.py "
            f"--group_path={group_path} "
            f"--result_path={result_path}"
        )
        
        jobs.append(command_reg)
    
    # Execute registration
    with multiprocessing.Pool(processes=2) as pool:
        pool.map(my_run, jobs)
    
    # Run again for those who is shut down by OS
    missing_jobs = list()
    for result_path in result_paths:
        if "eva_data" in os.listdir(result_path):
            continue
        group_name = os.path.basename(result_path)
        group_path = os.path.join(GOLDCASE_PATH, group_name)
        command_reg = (
            f"python register.py "
            f"--group_path={group_path} "
            f"--result_path={result_path}"
        )
        missing_jobs.append(command_reg)
    
    with multiprocessing.Pool(processes=2) as pool:
        pool.map(my_run, missing_jobs)
        
    return

def mode_evaluate():
    """
    Mode for evaluation
    """
    # Get path for long-term storage
    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d-%H%M%S")
    des_path = os.path.join(ARCHIVE_PATH, formatted_time)
    jobs = list()
    # Scan database
    for group_name in natsorted(os.listdir(EVALUATE_READY_PATH)):
        result_path = os.path.join(EVALUATE_READY_PATH, group_name)
        if not "eva_data" in os.listdir(result_path):
            continue
        jobs.append(result_path)
    
    with multiprocessing.Pool(processes=8) as pool:
        pool.map(evaluate, jobs)
    
    # Move them into archive
    print("Copying files to archive...")
    shutil.copytree(EVALUATE_READY_PATH, des_path)
    
    # Postprocess
    # pixel_error_mean, um_error_mean = postprocess(des_path)
    # print(f"Mean pixel error: {pixel_error_mean:.2f} pixels.")
    # print(f"Mean um error: {um_error_mean:.2f}um")
    
    return

def main():
    parser = argparse.ArgumentParser(description="Indicate parameters, use --help for help.")
    parser.add_argument("--mode", type=str, help="script mode")
    # Initialize parser
    args = parser.parse_args()
    mode = args.mode
    if mode == "register":
        mode_register()
    elif mode == "evaluate":
        mode_evaluate()
    else:
        raise Exception("Unsupported mode")
    
    return

if __name__ == "__main__":
    main()

    print("Program finished!")
