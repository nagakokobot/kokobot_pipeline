import pyzed.sl as sl
import cv2
import sys
import matplotlib.pyplot as plt


class Camera:
    #TODO: make the class to incorporate stereo depth and function calls in the init without passing args explicitly.
   

    def __init__(self,initparameters, runtimeparameters, save_path:str=None,show_workspace:bool = False):
        
        self.initpara = initparameters
        self.runtimepara = runtimeparameters
        self.s_path = save_path
        self.show_workspace = show_workspace

        self.zed = self.open_camera()
        try:
            s_number = self.zed.get_camera_information().serial_number
            print('Camera object created for the camera serial number:', s_number)
            self.bgra, self.depth = self.get_mats()
            self.rgb = cv2.cvtColor(self.bgra.get_data(), cv2.COLOR_BGRA2RGB)
        except Exception as e:
            print('closing the cam object because of the following error:')
            print(e)
            self.zed.close()
        try:
            if self.show_workspace:
                self.show_wrkspc(serial_num =s_number)
        except Exception as e:
            print('closing the cam object because of the following error:')
            print(e)
            self.zed.close()
        try:         
            if self.s_path != None: 
                self.rgb_path, self.depth_path = self.save_rgbd()
        except Exception as e:
            print('closing the cam object because of the following error:')
            print(e)
            self.zed.close()        
        cv2.destroyAllWindows()
        #zed.close()
            

    def open_camera(self):
        #TODO: Check the if condition, if there is any way to close the existing camera object and open new camera object
        cam_object = sl.Camera()
        err = cam_object.open(self.initpara)
        if err != sl.ERROR_CODE.SUCCESS:
            print(repr(err))
            cam_object.close()
            sys.exit()
        
        return cam_object
    
    def get_mats(self):
        rgb_mat = sl.Mat()
        depth_mat = sl.Mat()

        new_frame_err = self.zed.grab(self.runtimepara)
        if new_frame_err == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(rgb_mat, sl.VIEW.LEFT)
            self.zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)

        return rgb_mat, depth_mat
    
    def show_wrkspc(self, serial_num = int):
        '''
        Shows the rgb and depth image in a subplot
        Note: this stops the execution untill the window is closed.
        '''
        # Assuming rgb_np is the RGB image and depth_np is the depth image
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Display the RGB image
        axes[0].imshow(self.rgb)
        axes[0].set_title('RGB Image')
        axes[0].axis('off')  # Hide axis

        # Display the Depth image
        # Use a colormap to better visualize the depth data
        depth_img = axes[1].imshow(self.depth.get_data(), cmap='gray')
        axes[1].set_title('Depth Image')
        axes[1].axis('off')  # Hide axis
        fig.colorbar(depth_img, ax=axes[1], orientation='vertical', fraction=0.046, pad=0.04)
        fig.canvas.manager.set_window_title(f'Camera Workspace - Serial: {serial_num}')
        plt.tight_layout()
        plt.savefig(f'{self.s_path}/workspace_view.png')
        plt.show()        

        pass

    def save_rgbd(self):
        img_path = self.s_path + '/rgb_image.png'
        d_path = self.s_path+'/true_depth.tiff'

        plt.imsave(img_path, self.rgb)
        plt.imsave(d_path, self.depth.get_data(), cmap='gray')
        print('Images of the workspace saved')
  
        return img_path, d_path
    
    def close_cam(self):
        if self.zed.is_opened():
            self.zed.close()
        pass


if __name__ == '__main__': 
    
    from camera_parameters import get_init_camera_paramaters, get_runtime_camera_parameters
    from helpers import create_folder

    args = {'camera_resolution': 'HD1080'}
    #          ,'camera_fps': 60,
    #      'depth_mode': 'PERFORMANCE'}
    _, s_path = create_folder('test_folder1')

    init = get_init_camera_paramaters(args=args, save_path=s_path) 
    run = get_runtime_camera_parameters(args = args, save_path=s_path)

    cam = Camera(initparameters= init, runtimeparameters=run, show_workspace= True, save_path=s_path)
    cam.close_cam