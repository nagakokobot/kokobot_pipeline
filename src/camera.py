import pyzed.sl as sl
import cv2


class Camera:
    #TODO: make the class to incorporate stereo depth and function calls in the init without passing args explicitly.
   

    def __init__(self,initparameters, runtimeparameters, save_path,show_workspace:bool = False):
        
        self.initpara = initparameters
        self.runtimepara = runtimeparameters
        self.s_path = save_path
        self.show_workspace = show_workspace

        self.zed = self.open_camera()
        s_number = self.zed.get_camera_information().serial_number
        print('Camera object created for the camera serial number:', s_number)
        self.rgb, self.depth = self.get_mats()
        if self.show_workspace:
            named_window = f'workspace_{s_number}'
            cvImage = self.rgb.get_data()
            cv2.imshow(named_window, cvImage)
            #print('press any to exit and continue execution')
            cv2.waitKey(4000)
        self.rgb_path, self.depth_path = self.save_rgbd(self.rgb, self.depth, self.s_path)
        
        cv2.destroyAllWindows()
        #zed.close()
            

    def open_camera(self):
        cam_object = sl.Camera()
        err = cam_object.open(self.initpara)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            cam_object.close()
            exit()
        
        return cam_object
    
    def get_mats(self):
        rgb_mat = sl.Mat()
        depth_mat = sl.Mat()

        new_frame_err = self.zed.grab(self.runtimepara)
        if new_frame_err == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(rgb_mat, sl.VIEW.LEFT)
            self.zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

        return rgb_mat, depth_mat

    def save_rgbd(self, rgb_mat, depth_mat, save_path):
        img_path = save_path + '/rgb_image.png'
        d_path = save_path+'/true_depth.tiff'

        cv2.imwrite(img_path, self.rgb.get_data())
        print('saved rgb image')
        cv2.imwrite(d_path, self.depth.get_data())
        print('saved true_depth.tiff')
  
        return img_path, d_path
    
    def close_cam(self):
        if self.zed.is_opened():
            self.zed.close()
        pass
