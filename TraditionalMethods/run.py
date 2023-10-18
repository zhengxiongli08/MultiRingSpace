
# This is a python script. Used to automatically generate results for traditional methods.


import os
import shutil
import subprocess
import datetime


# Global variables
MONOMER_PATH = "/mnt/Disk1/whole_slide_image_analysis/Lizhengxiong/Projects/MultiRingSpace/BiopsyDatabase/monomer"
POLYSOME_PATH = "/mnt/Disk1/whole_slide_image_analysis/Lizhengxiong/Projects/MultiRingSpace/BiopsyDatabase/polysome"
RESULT_PATH = "/mnt/Disk1/whole_slide_image_analysis/Lizhengxiong/Projects/MultiRingSpace/result2"
ARCHIVE_PATH = "/mnt/Disk1/whole_slide_image_analysis/Lizhengxiong/Projects/MultiRingSpace/archive"

def my_run(command: str):
    """
    A warp for subprocess.check_call function
    """
    command_new = command.split()
    subprocess.check_call(command_new)

    return

def execute_algorithm(algorithm_type, archive_folder_path):
    """
    Execute corresponding algorithm.
    algorithm_type options:
        akaze
        brisk
        kaze
        orb
        sift
    """
    # Create algorithm results folder
    algorithm_folder_path = os.path.join(archive_folder_path, algorithm_type)
    # Create path string for monomer and polysome group
    monomer_result_path = os.path.join(algorithm_folder_path, "monomer")
    polysome_result_path = os.path.join(algorithm_folder_path, "polysome")

    # For each monomer data group, send it to algorithm program.
    for monomer_group in os.listdir(MONOMER_PATH):
        # Remove the result folder and create a new one
        if os.path.exists(RESULT_PATH):
            shutil.rmtree(RESULT_PATH)
        os.mkdir(RESULT_PATH)
        # Get path to store monomer group's results
        monomer_group_path = os.path.join(MONOMER_PATH, monomer_group)
        # Execute the registration program
        my_run(f"python {algorithm_type}.py --group_path={monomer_group_path} --slide_type=monomer")
        # Create archive folder for this group of data
        monomer_group_result_path = os.path.join(monomer_result_path, monomer_group+"-result")
        # Copy results from result folder to archive folder
        shutil.copytree(RESULT_PATH, monomer_group_result_path)
        
    # For each polysome data group, send it to algorithm program
    for polysome_group in os.listdir(POLYSOME_PATH):
        # Remove the result folder and create a new one
        if os.path.exists(RESULT_PATH):
            shutil.rmtree(RESULT_PATH)
        os.mkdir(RESULT_PATH)
        # Get path to store monomer group's results
        polysome_group_path = os.path.join(POLYSOME_PATH, polysome_group)
        # Execute the registration program
        my_run(f"python {algorithm_type}.py --group_path={polysome_group_path} --slide_type=polysome")
        # Create archive folder for this group of data
        polysome_group_result_path = os.path.join(polysome_result_path, polysome_group+"-result")
        # Copy results from result folder to archive folder
        shutil.copytree(RESULT_PATH, polysome_group_result_path)
    
def main():
    # Get current time
    current_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    # Create a folder in archive directory
    archive_folder_path = os.path.join(ARCHIVE_PATH, current_time)
    os.mkdir(archive_folder_path)
    
    # Execute every algorithm in order
    execute_algorithm("akaze", archive_folder_path)
    execute_algorithm("brisk", archive_folder_path)
    execute_algorithm("kaze", archive_folder_path)
    execute_algorithm("orb", archive_folder_path)
    execute_algorithm("sift", archive_folder_path)
    
            
    return

if __name__ == "__main__":
    main()
