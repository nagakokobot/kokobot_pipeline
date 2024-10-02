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
        flag = True
    else:
        print(f'overwriting the folder {name}, which already exists')

    return flag, c_path

if __name__ == '__main__':
     
    create_folder(name = 'now')
