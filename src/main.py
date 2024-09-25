import argparse
import sys
import pyzed.sl as sl
from datetime import datetime

from camera_parameters import get_camera_serial_number, get_camera_paramaters
from helpers import create_folder


def parse_args():

    parser = argparse.ArgumentParser(description="Run the pipeline with specified parameters.")
    
    #parser.add_argument('--name', type=str, required=True, help="Name parameter (mandatory).")
    parser.add_argument('--camera-resolution', type=str, help="Name parameter (mandatory).")
    #no parsing arguments for now.

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    #for testing:
    #sys.argv = ['main.py', '--camera-resolution', 'HD1080']

    #main script runs here...
    args = parse_args()

    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    flag, s_path = create_folder(folder_name)
    if flag:
        # Check if cameras were connected 
        cameras,serial_number = get_camera_serial_number()

        get_camera_paramaters(args = args, serial_number= serial_number, save_path = s_path)