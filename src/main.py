import argparse
import pyzed.sl as sl

from camera_parameters import get_camera_serial_number



def parse_args():

    parser = argparse.ArgumentParser(description="Run the pipeline with specified parameters.")
    
    #parser.add_argument('--name', type=str, required=True, help="Name parameter (mandatory).")
    #no parsing arguments for now.

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    
    args = parse_args()

# Check if cameras were connected 
    cameras,serial_number = get_camera_serial_number()