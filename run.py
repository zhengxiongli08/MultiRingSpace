
# Automatically run gold cases in parallel

import os
import shutil
import subprocess
import multiprocessing
from datetime import datetime
from evaluate import evaluate
from natsort import natsorted


# Global constants
GOLDCASE_PATH = "../BiopsyDatabase/WSI_100Cases"
EVALUATE_READY_PATH = "../archive/eva_ready"

# Functions
def my_run(command):
    subprocess.run(command, shell=True)

    return

def main():
    # Parameters for registration    
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
    with multiprocessing.Pool(processes=10) as pool:
        pool.map_async(my_run, commands_reg)
        pool.close()
        pool.join()
    
    # Execute evaluation
    with multiprocessing.Pool(processes=10) as pool:
        pool.map_async(my_run, commands_eva)
        pool.close()
        pool.join()
    
    return

if __name__ == "__main__":
    main()

    print("Program finished!")
