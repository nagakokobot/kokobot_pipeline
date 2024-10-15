import pyzed.sl as sl
import os
import numpy






def camera_init_parameters():

#TODO: add more options for coordinate transformations

    d_init_params = {'camera_resolution':{'HD720':sl.RESOLUTION.HD720},
                     'camera_fps': 0,
                     'depth_mode':{'ULTRA':sl.DEPTH_MODE.ULTRA},
                     'coordinate_units':{'mm':sl.UNIT.MILLIMETER},
                     'coordinate_system':{'IMAGE':sl.COORDINATE_SYSTEM.IMAGE},
                     'depth_minimum_distance':-1,
                     }

    parms_mapping = {'camera_resolution': {'HD1080': sl.RESOLUTION.HD1080},
                        'depth_mode': {'PERFORMANCE': sl.DEPTH_MODE.PERFORMANCE,
                                       'QUALITY': sl.DEPTH_MODE.QUALITY},
                        'coordinate_units':{'cm':sl.UNIT.CENTIMETER,
                                            'm': sl.UNIT.METER},
                        'coordinate_system': {}}
    
    parms_options = {'camera_resolution': ['HD720', 'HD1080'],
                     'camera_fps': [60,30,15],
                     'depth_mode': ['ULTRA', 'PERFORMANCE', 'QUALITY'],
                     'coordinate_units': ['mm','cm', 'm']}

    return d_init_params, parms_mapping, parms_options


def camera_runtime_parameters():

    params = {'measure3D_reference_frame': {'world': sl.REFERENCE_FRAME.WORLD,
                                                    'camera': sl.REFERENCE_FRAME.CAMERA},
                      'confidence_threshold': range(1,101),
                      'texture_confidence_threshold': range(1,101)}

    #TODO: delete the dict later and change it to the list below. (change the helper.sort_args to accept options as a list also before changing this function)
    #params = ['measure3D_reference_frame','confidence_threshold','texture_confidence_threshold']
    return params


def get_model_path(task:str, version: str = 'v5'):

    '''
    Ensure that the pre trained models exists in the folder pre_trained_model. else, the excution stops.
    - If the pre trained models exist in the above folder and the function still raises FileNotFoundError, check the names
      of the saved models. The function is hard coded to the names: 'best.pt' for detection task
    -Folder structure: 
    ├── pre_trained_models 
    │   ├── detection models 
    │        ├──v5
    │            ├──best.pt
    │        └──v8 
    │             ├──best.pt           
    │   └──grasp models 
    
    '''

    #check for the path of pre trained model folder first
    parent_path = 'pre_trained_models'
    path = ''
    #os.path.exists(parent_path)
    if not os.path.exists(parent_path):
        raise FileNotFoundError(f"The directory {parent_path} does not exist.")
    folders = os.listdir('pre_trained_models') 
    if task == 'detection' and 'detection' in folders:
        path = os.path.join(parent_path, 'detection', 'v5', 'best.pt')
        if version == 'v8':
            path = os.path.join(parent_path, 'detection', 'v8', 'best.pt')
    #TODO: change the model name for the grasp synthesis when working on grasp generation
    if task == 'grasp_synthesis' and 'grasp_synthesis' in folders:
        path = os.path.join(parent_path, 'grasp_synhesis', 'model')

    if path == '':
        raise FileNotFoundError('No path can be found for getting pre trained model, check all the saved model names are correct and in correct folders')
    if not os.path.exists(path) and path != '':
        raise NameError(f'No path found "{path}" for {task} task')
    return path




if __name__ == '__main__':

    path = get_model_path(task='detection', version='v5')
    print(type(path), path)