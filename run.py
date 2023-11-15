
# Automatically run gold cases and evaluate the results

import os
import shutil
import subprocess
from datetime import datetime
from evaluate import evaluate
from natsort import natsorted
from logger import Logger


# Global constants
GOLDCASE_PATH = "../BiopsyDatabase/WSI_100Cases"
RESULT_PATH = "../result"
ARCHIVE_PATH = "../archive"
EVALUATE_READY_PATH = os.path.join(ARCHIVE_PATH, "evaluate_ready")

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
    # Get path to restore results without evaluation
    no_evaluation_path = EVALUATE_READY_PATH
    if os.path.exists(no_evaluation_path):
        shutil.rmtree(no_evaluation_path)
    os.makedirs(no_evaluation_path)
    # Get path for archive
    archive_folder_path = os.path.join(ARCHIVE_PATH, current_time)
    
    # Do it for gold case dataset without evaluation
    group_names = natsorted(os.listdir(GOLDCASE_PATH))
    for group_name in group_names:
        # Remove the result folder and create a new one
        if os.path.exists(RESULT_PATH):
            shutil.rmtree(RESULT_PATH)
        os.mkdir(RESULT_PATH)
        my_run(f"python main.py \
                --groups_path={GOLDCASE_PATH} \
                --group_name={group_name} \
                --result_path={RESULT_PATH} \
                --slide_type=monomer")
        # Get path for case's result
        group_result_path = os.path.join(no_evaluation_path, group_name+"-result")
        # Copy results from result folder to archive folder
        shutil.copytree(RESULT_PATH, group_result_path)
    
    # Evaluate the results and record errors
    result_names = natsorted(os.listdir(EVALUATE_READY_PATH))
    errors = list()
    for result_name in result_names:
        result_path = os.path.join(EVALUATE_READY_PATH, result_name)
        mean_error = evaluate(result_path)
        errors.append(mean_error)
    # Move results to archive folder
    shutil.move(EVALUATE_READY_PATH, archive_folder_path)
    
    # Print results and save it in .txt file
    mean_error = sum(errors) / len(errors)
    result_str = f"Mean error for {len(result_names)} cases: {mean_error}."
    txt_path = os.path.join(archive_folder_path, "error.txt")
    with open(txt_path, 'w') as file:
        file.write(result_str)
    print(result_str)
    
    return

def single_run():
    """
    Run a single case
    """
    # Remove the result folder and create a new one
    if os.path.exists(RESULT_PATH):
        shutil.rmtree(RESULT_PATH)
    os.mkdir(RESULT_PATH)
    # Run registration
    group_name = "TM-2-40magnification-group3"
    my_run(f"python main.py \
            --groups_path={GOLDCASE_PATH} \
            --group_name={group_name} \
            --result_path={RESULT_PATH}")
    # Evaluate results
    mean_error = evaluate(RESULT_PATH)
    print(f"Mean error: {mean_error:.3f} pixels.")
    
    return
    
if __name__ == "__main__":
    # main()
    
    single_run()
    
    print("Program finished!")
