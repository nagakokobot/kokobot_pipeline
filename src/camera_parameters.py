import pyzed.sl as sl




'''
This script is used to 
- get the camera status
- get the initial parameters of the cameras connected
- get the errors if there are any with the connected cameras
- get and save the images of the work space when the operator is ready (rgb-d, depth, stereo depth, true depth)
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

    
