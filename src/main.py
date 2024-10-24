import argparse
import sys
import pyzed.sl as sl
import torch
from datetime import datetime

from camera_parameters import get_camera_serial_number, get_init_camera_paramaters, get_runtime_camera_parameters
from helpers import create_folder
from camera import Camera
from detection import detection, run_inference


def parse_args():

    parser = argparse.ArgumentParser(description="Run the pipeline with specified parameters.")
    
    #parser.add_argument('--name', type=str, required=True, help="Name parameter (mandatory).")
    #for init_parameters
    parser.add_argument('--camera-resolution', type=str, default= 'HD720', choices=['HD720', 'HD1080'], help="Name parameter (mandatory).")
    #parser.add_argument('--fps', type=int, choices=[15,30,60], help = 'Frames per second required (depends on the resolution)') #TODO: fix the problem with argparse

    #for runtime_parameters
    #parser.add_argument('--measure3D-reference-frame', type=str, default= 'camera', choices=['world', 'camera'], help = 'reference frame for next camera frame')
    #parser.add_argument('--confidence-threshold', type=int, default= 95, choices= range(1,101), help = 'removes the depth values from the mat if depth confidence less than this parameter')
    #parser.add_argument('--texture-confidence-threshold', type=int, default= 100, choices= range(1,101), help = 'Decreasing this value will remove depth data from image areas which are uniform')

    #for detection.py
    parser.add_argument('--det_device', type=str, default= 'cpu', choices=['cpu', 'gpu'], help="select the device where the detection happens")
    parser.add_argument('--det_ver', type=str, default= 'v5', choices=['v5', 'v8'], help="select the version of yolo to be used for detection")


    args = parser.parse_args()

    return args


if __name__ == '__main__':

    #for testing:
    sys.argv = ['main.py', '--camera-resolution', 'HD720']
    #sys.argv = ['main.py', '--fps', 30]
    #sys.argv = ['main.py', '--measure3D-reference-frame', 'world']
    #sys.argv = ['main.py', '--confidence-threshold', 50]
    #sys.argv = ['main.py', '--texture-confidence-threshold', 50]
    


    ##main script runs here...

    #get the args as dict for passing to the functions. makes it easy for testing and clear code
    args = parse_args()
    print('args initially:', args)
    if not isinstance(args, dict):    # for testing purpose
        args_dict = vars(args)
    else:
        pass

    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    flag, s_path = create_folder(folder_name)
    if flag:
        # Check if cameras were connected 
        cameras,serial_number = get_camera_serial_number()

        init_params = get_init_camera_paramaters(args = args_dict, serial_number= serial_number, save_path = s_path)
        runtime_params = get_runtime_camera_parameters(args = args_dict, save_path=s_path)
        cam = Camera(initparameters= init_params, runtimeparameters=runtime_params, save_path= s_path, show_workspace= True)
        cam_obj = cam.zed
        rgb, depth, rgb_path, depth_path = cam.rgb, cam.depth, cam.rgb_path, cam.depth_path
        #close the camera object
        if cam_obj.is_opened():
            cam.close_cam()

        model = detection(workfolder = folder_name, detector_version = args.det_ver)

    
        ## TODO: test inference time on cpu vs gpu:
        
        # in progress
        # Get inference on Images from the work folder
        # run Inference on rgb_image 's in the s_path
        detections_dataframe = run_inference(image_path= rgb_path, device = args.det_device, model = model.model)
        

