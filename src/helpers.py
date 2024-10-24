import os
from datetime import datetime

import pyzed.sl as sl


from defaults import camera_init_parameters    #delete this line later


def sort_args(args, options):

    args_keys = set(args.keys())
    options_keys = set(options.keys())
    possible_options = args_keys.intersection(options_keys)
    return possible_options

def create_folder(name):
    #later has to be changed to handle better folder creation and check
    flag = False
    c_path = f'project_aux/{name}'
    if not os.path.exists(c_path):
        os.mkdir(c_path)
        print(f'Working folder is created with the name: {c_path}')
        flag = True
    else:
        print(f'overwriting the folder {name}, which already exists')

    return flag, c_path


def get_model_path(task:str, version: str = 'v5'):

    '''
    Ensure that the pre trained models exists in the folder pre_trained_model. else, the excution stops.
    - If the pre trained models exist in the above folder and the function still raises FileNotFoundError, check the names
      of the saved models. The function is hard coded to the names: 'best.pt' for detection task
    - TODO: make the function take the user defined name for the model and checks in the folder for the name.
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
