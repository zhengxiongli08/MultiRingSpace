
# Automatically run all data group

import os
import shutil
import subprocess


# Functions
def my_run(command: str):
    """
    A warp for subprocess.check_call function
    """
    command_new = command.split()
    subprocess.check_call(command_new)

    return

