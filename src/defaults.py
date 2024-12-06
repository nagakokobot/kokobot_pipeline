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
                     'depth_minimum_distance':1000,
                     'depth_maximum_distance': 1500
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

def det_args():

    options = {'det_conf':0.8,
               'iou':0.7,
               'img_sz': None
               }

    return options


if __name__ == '__main__':

    _,_,options = camera_init_parameters()
    print(type(options), options)