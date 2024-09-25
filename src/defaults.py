import pyzed.sl as sl
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