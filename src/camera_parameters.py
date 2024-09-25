import pyzed.sl as sl
import argparse
import os

from defaults import camera_init_parameters
from helpers import sort_args, create_folder



'''
This script is used to 
- get the camera status
- get the initial parameters of the cameras connected
- get the errors if there are any with the connected cameras
- get and save the images of the work space when the operator is ready (rgb-d, depth, stereo depth, true depth)

# 35162414 - is the camera with birds eye view
'''





def save_rgb_image():
    pass


def save_depth_images():
    pass


def get_camera_serial_number():

    cameras = sl.Camera.get_device_list()
    if cameras is None:
        print('No cameras were detected')
        # The whole python execution has to stop here.
        exit()
    else:
        if len(cameras)== 1:
            print('1 camera found')
            serial_number = cameras[0].serial_number
        else :
            print('Two cameras found')
            cam_dict = {}
            for cam in range(len(cameras)):
                cam_dict[cam] = cameras[cam].serial_number
            print(cam_dict)
            while True:
                choice = int(input('enter the id of camera to be used (0,1..)'))
                if choice in cam_dict:
                    serial_number = cam_dict[choice]
                    break
                else:
                    print('your choice is invalid, choose in range 0 to {0}'.format(len(cameras)-1))


    return cameras, serial_number

def get_camera_paramaters(args:argparse.ArgumentParser, serial_number:int = 35162414, save_path:str=None):
    '''
    The camera parameters are required for opening the camera object.
    - The initparameters contains many important features on how to open the camera.
    - The specific init file has to be saved in the project aux folder so that the camera can be opened with same 
    specifications later in the pipeline.
    - min depth distance, depth mode
    - coordinate systems, coordinate units
    -camera fps, camera resoultion
    '''
    #get the init_parameters, if no arguments is passed return default initparameters
    init_params = sl.InitParameters()
    init_params.set_from_serial_number(serial_number = serial_number)

    ## get the user requested parameters and fill the new_args_dict.
    d_init_params, mapping, options  = camera_init_parameters()

    if not isinstance(args, dict):    # for testing purpose
        args = vars(args)
    print(args)  #TODO: Remove the none values from args
    possible_args = sort_args(args, options)
    #print('possible args before fps :', possible_args)

    if possible_args:
        new_args_dict = {}        
        #TODO: also update the logic to handle coordinate system and minimum depth distance
        if 'camera_fps' in possible_args:   
            if 'camera_resolution' not in possible_args:
                new_args_dict['camera_fps'] = args['camera_fps']
            else:
                if args['camera_resolution'] == 'HD720':
                    new_args_dict['camera_fps'] = args['camera_fps']
                elif args['camera_resolution'] == 'HD1080' and args['camera_fps']>30:
                    print('Cannot use fps more than 30 with HD1080 resolution. Setting fps to 30')
                    new_args_dict['camera_fps'] = 30
                elif args['camera_resolution'] == 'HD1080' and args['camera_fps']<30:
                    new_args_dict['camera_fps'] = args['camera_fps']
            possible_args.remove('camera_fps')
        #print('possible args after fps :', possible_args)
        #print('new args dict after fps',new_args_dict)

        for i in possible_args:
            if args[i] in d_init_params[i]:
                new_args_dict[i] = d_init_params[i][args[i]]
            if args[i] in mapping[i]:
                new_args_dict[i] = mapping[i][args[i]]
        #print('new args dict after all args: ', new_args_dict)
        ## set the initparameters acc to the new_args_dict
        for i, y in new_args_dict.items():
            if i == 'camera_resolution':
                init_params.camera_resolution = y
            elif i == 'camera_fps':
                init_params.camera_fps = y
            elif i == 'depth_mode':
                init_params.depth_mode = y
            elif i == 'coordinate_units':
                init_params.coordinate_units = y

    init_params.save(save_path+"/initParameters.conf")
    return init_params








if __name__ == '__main__':


    u_args = {'camera_resolution': 'HD720',
              'camera_fps': 60,
          'depth_mode': 'PERFORMANCE'}
    #u_args = {'runtime': 0}
    retu = get_camera_paramaters(args = u_args)

    