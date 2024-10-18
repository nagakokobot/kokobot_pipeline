import pyzed.sl as sl
import cv2


class get_images:
    #TODO: make the class to incorporate stereo depth.
   

    def __init__(self,initparameters, runtimeparameters, save_path,show_workspace:bool = False):
        
        self.initpara = initparameters
        self.runtimepara = runtimeparameters
        self.s_path = save_path
        self.show_workspace = show_workspace

        zed = self.open_camera(self.initpara)
        s_number = zed.get_camera_information().serial_number
        print('Camera object created for the camera serial number:', s_number)
        rgb, depth = self.get_mats(zed,self.runtimepara)
        if self.show_workspace:
            named_window = f'workspace_{s_number}'
            cvImage = rgb.get_data()
            cv2.imshow(named_window, cvImage)
            #print('press any to exit and continue execution')
            cv2.waitKey(4000)
        self.save_rgbd(rgb, depth, self.s_path)
        
        cv2.destroyAllWindows()
        zed.close()
            

    def open_camera(self, initparams):
        cam_object = sl.Camera()
        err = cam_object.open(initparams)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            cam_object.close()
            exit()
        
        return cam_object
    
    def get_mats(self, cam_object,runtimeparams):
        rgb_mat = sl.Mat()
        depth_mat = sl.Mat()

        new_frame_err = cam_object.grab(runtimeparams)
        if new_frame_err == sl.ERROR_CODE.SUCCESS:
            cam_object.retrieve_image(rgb_mat, sl.VIEW.LEFT)
            cam_object.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

        return rgb_mat, depth_mat

    def save_rgbd(self, rgb_mat, depth_mat, save_path):
        img_path = save_path + '/rgb_image.png'
        d_path = save_path+'/true_depth.tiff'

        cv2.imwrite(img_path, rgb_mat.get_data())
        print('saved rgb image')
        cv2.imwrite(d_path, depth_mat.get_data())
        print('saved true_depth.tiff')
  
        return img_path, d_path
