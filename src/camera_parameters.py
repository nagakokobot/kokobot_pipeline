import pyzed.sl as sl
import argparse
import os

from defaults import camera_init_parameters, camera_runtime_parameters
from helpers import sort_args, create_folder



'''
This script is used to 
- get the camera status
- get the initial parameters of the cameras connected
- get the errors if there are any with the connected cameras

# 35162414 - is the camera with birds eye view
'''

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

def get_init_camera_paramaters(args:dict, save_path:str = False,serial_number:int = 35162414):
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

    #print(args)  #TODO: Remove the none values from args
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
    if save_path:
        init_params.save(save_path+"/initParameters")
    return init_params

def get_runtime_camera_parameters(args:dict, save_path:str):

    runtime_params = sl.RuntimeParameters()
    d_runtime_params = camera_runtime_parameters()
    possible_args = sort_args(args, d_runtime_params)
    if possible_args:
        for i in possible_args:
            if i == 'measure3D_reference_frame' and args[i] != 'camera':   #TODO: add all the possible options in the argprase
                runtime_params.measure3D_reference_frame = sl.REFERENCE_FRAME.WORLD
            elif i == 'confidence_threshold' and args[i] != 95:
                runtime_params.confidence_threshold = args[i]
            elif i == 'texture_confidence_threshold':
                runtime_params.texture_confidence_threshold = args[i]

    try: #important while testing as save wont overwrite .conf file for run time parameters
        s_flag = runtime_params.save(save_path + '/runtimeParameters')
        if not s_flag:
            os.remove(save_path+'/runtimeParameters.yml')
            runtime_params.save(save_path+ '/runtimeParameters')
    except Exception as e:
        print(e)
    
    return runtime_params





if __name__ == '__main__':

    '''
    #for testing get_init_camera_parameters..
    u_args = {'camera_resolution': 'HD720',
              'camera_fps': 60,
          'depth_mode': 'PERFORMANCE'}
    _, s_path = create_folder('test_folder1')
    #u_args = {'runtime': 0}
    init_params = get_init_camera_paramaters(args = u_args, save_path=s_path)
    print(init_params)
    '''

    #for testing get_runtime_camera_parameters..
    _, s_path = create_folder('test_folder1')
    u_args = {'measure3D_reference_frame': 'world',
              'confidence_threshold': 70}
    run_params = get_runtime_camera_parameters(args = u_args, save_path=s_path)
    
    