import argparse
import sys
sys.path.append('/home/student/naga/kokobot_pipeline')  #to run from terminal
import pyzed.sl as sl
import pandas as pd
from datetime import datetime
import os, shutil
import numpy as np

from camera_parameters import get_camera_serial_number, get_init_camera_paramaters, get_runtime_camera_parameters
from helpers import create_folder
from camera import Camera
from detection import Detector, Inference
from grasp_synthesis import load_grasping_model, Process_crops, pred_grasps_and_display, make_tensors_and_predict_grasps, display_grasps_per_object, display_grasps_per_image
from segment import get_masks_from_seg_res
from transform import Robot_coordinates

os.environ['YOLO_VERBOSE'] = 'False'


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
    #parser.add_argument('--det_conf', type=str, choices=range(0.01, 1), help="detection model confidence threshold")
    #parser.add_argument('--iou', type=str, choices=range(0.01,1), help="threshold for Non-Maximum Suppression (NMS)")
    #parser.add_argument('--img_sz', type=str, default= 'v5', choices=['v5', 'v8'], help="select the version of yolo to be used for detection")

    #for grasping
    parser.add_argument('--grasp_model_name', type=str, help="Name of the model choosen to use for grasp synthesis")


    args = parser.parse_args()

    return args


if __name__ == '__main__':

    #for testing:
    #sys.argv = ['main.py', '--camera-resolution', 'HD720']
    #sys.argv = ['main.py', '--fps', 30]
    #sys.argv = ['main.py', '--measure3D-reference-frame', 'world']
    #sys.argv = ['main.py', '--confidence-threshold', 50]
    #sys.argv = ['main.py', '--texture-confidence-threshold', 50]
    
    #till the argparse is sorted:
    args_dict = {'camera_resolution': 'HD1080',
                 'coordinate_units': 'mm',
            'det_conf': 0.75,
            'iou': 0.7,
            'det_ver':'v8',
            'det_device':'cpu',
            'grasp_model_name': 'd_grasp.pt',
            'include_segmentation':True}

    ##main script runs here...

    #get the args as dict for passing to the functions. makes it easy for testing and clear code
    if not args_dict:
        args = parse_args()
        print('args initially:', args)
        if not isinstance(args, dict):    # for testing purpose
            args_dict = vars(args)
        else:
            pass

    current_time = datetime.now()
    folder_name = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    flag, s_path = create_folder(sub_name=folder_name, parent_name='project_aux')
    if flag:
        # Check if cameras were connected 
        print('1: Getting camera serial numbers to open camera and get images')
        cameras,serial_number = get_camera_serial_number()

        init_params = get_init_camera_paramaters(args = args_dict, serial_number= serial_number, save_path = s_path)
        runtime_params = get_runtime_camera_parameters(args = args_dict, save_path=s_path)
        cam = Camera(initparameters= init_params, runtimeparameters=runtime_params, save_path= s_path, show_workspace= False)
        cam_obj = cam.zed
        rgb, depth_cam, rgb_path, depth_path = cam.rgb, cam.depth.copy(), cam.rgb_path, cam.depth_path
        
        #model = Detector(detector_version = args_dict['det_ver'], device=args_dict['det_device'])
        print('2: Loading detector for object detection for inference on Images')
        with Detector(detector_version = args_dict['det_ver'], device=args_dict['det_device']) as d_model:
            model = d_model.model

        inf1 = Inference(model = model, image= rgb.copy(), detector_version= args_dict['det_ver'], project = s_path,args = args_dict)
        pr = inf1.process_results()
        d_model.del_model()
        if pr.empty:
            print('No objects found in the workspace, deleting work folder')
            shutil.rmtree(s_path)
            sys.exit()
        else:
            print('Detected Objects in the work space are: ',pd.Series(pr.loc[:,'name']).to_list())
        
        
        print('3: Loading grasp model and croping the image for generating individual grasps')
        grasp_model = load_grasping_model(model_name=args_dict['grasp_model_name'])
        object_crops = Process_crops(rgb=rgb, depth=depth_cam, coordinates=pr, include_segmentation = True)
        image_dict = object_crops.image_dict
        #object_crops.show_crops()

        if not args_dict['include_segmentation']:
            image_dict, grasps = pred_grasps_and_display(model = grasp_model, image_dict= image_dict, display_images= True)
        else:
            #for i, (obj, itype) in enumerate(img_dict.items()):
            image_dict = get_masks_from_seg_res(image_dict= image_dict)
            image_dict = make_tensors_and_predict_grasps(image_dict=image_dict, grasp_model=grasp_model)
            
            display_grasps_per_object(image_dict=image_dict)
        display_grasps_per_image(image_dict = image_dict, org_image = rgb)

        #get the saved transformation matrix and transform the coordinates of the grasp centers to robot coordinates
        t_m = Robot_coordinates()
        print('robot coordinates------')
        for obj,itype in image_dict.items():
            print(f'for object {obj} ')
            grasps = itype['grasps']
            for stype, im in grasps.items():
                i_grasp = im['image_grasps']
                yi, xi = i_grasp[0].center    #take the best grasp for all the segmentation types
                print(xi, yi)
                xc,yc,zc = cam.get_xyz(int(xi), int(yi))
                print(xc,yc,zc)
                if xc:
                  xr,yr,zr = t_m.transform_camera_to_robot([xc,yc,zc])
                else:
                    xr,yr,zr = None, None, None
                print(f'with seg_mask named {stype}, the robot coordinates are: {xr,yr,zr} in mm along x,y,z')
        cam.close_cam()

        #close the camera object
        #if cam_obj.is_opened():
        #    cam.close_cam()
