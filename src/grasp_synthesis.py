import os
import torch
from typing import Union
import numpy as np
import pandas as pd
from imageio.v2 import imread
import matplotlib.pyplot as plt
from ggcnn.models.ggcnn2 import GGCNN2

from PIL import Image
#for filling the nan and inf values
from scipy import interpolate
from scipy.ndimage import generic_filter, gaussian_filter

from helpers import get_model_path
from ggcnn.utils.dataset_processing.image import Image, DepthImage
from ggcnn.models.common import post_process_output
from ggcnn.utils.dataset_processing.evaluation import plot_output
from ggcnn.utils.dataset_processing.grasp import detect_grasps


#Loading the model(function)
def load_grasping_model(input_channels:int = 4, model_name:str = None):
    
    model_path = get_model_path(task='grasp_synthesis', model_name= model_name)
    model = GGCNN2(input_channels=input_channels)
    model.load_state_dict(torch.load(model_path))
    return model


class Process_crops:
    '''
    To process the obtained image or numpy array before passing to grasp synthesis model
    rgb: rgb format of the Image either a str to the path or numpy array
    depth: path to depth.tiff or numpy array of the depth image
    coordinates (optional): Path to the csv file generated, which containes the coordinates or pd.dataframe containing the detections
    '''
    #image_dict = {}
    def __init__(self,rgb:Union[str, np.ndarray], depth:Union[str, np.ndarray], coordinates: Union[str, pd.DataFrame]):
        self.rgb = rgb
        self.depth = depth
        self.coordinates = coordinates
        if isinstance(self.rgb, str):
            self.rgb = imread(self.rgb)
        if isinstance(self.depth, str):
            self.depth = imread(self.depth)
        if isinstance(self.coordinates, str):
            self.coordinates = pd.read_csv(self.coordinates, header=0,index_col=0)
        #self.image_dict.clear()
        self.image_dict = self.get_image_crops()
        #print(self.image_dict)
    def get_image_crops(self):
        image_dict = {}

        no_of_objects = len(self.coordinates.index)
        if no_of_objects>0:
            for i in range(no_of_objects):
                    obj = f'object_{i+1}_{self.coordinates.iloc[i,6]}'
                    #rgb_i = obj+'_rgb_i'
                    #depth_i = obj+'_depth_i' 
                    (xmin, ymin, xmax, ymax) = (int(self.coordinates.iloc[i,0]), 
                                                int(self.coordinates.iloc[i,1]), 
                                                int(self.coordinates.iloc[i,2]), 
                                                int(self.coordinates.iloc[i,3]))
                    rgb_i = Image(self.rgb)
                    depth_i = DepthImage(self.depth)
                    self.process_image(image = rgb_i, tup = (xmin, ymin, xmax, ymax))
                    self.process_image(image = depth_i, tup = (xmin, ymin, xmax, ymax))
                    #rgb_image_i = self.process_image(image = rgb_i, tup = (xmin, ymin, xmax, ymax))
                    #depth_image_i = self.process_image(image = depth_i, tup = (xmin, ymin, xmax, ymax))
                    image_dict[obj] = {}
                    image_dict[obj].update([('rgb_object', rgb_i),('depth_object', depth_i)])
                    #image_dict[obj].update([('rgb', rgb_image_i),('depth', depth_image_i)])
                    #plt.imshow(self.rgb)
                    #plt.show()
                    #tensor_i = self.make_tensors(fin_rgb=rgb_image_i, fin_depth= depth_image_i)
                    tensor_i = self.make_tensors(fin_rgb=rgb_i.img, fin_depth= depth_i.img)
                    image_dict[obj].update([('tensor', tensor_i)])
                    del rgb_i, depth_i
        #print(no_of_objects)
        return image_dict
    

    def process_image(self, image, tup:tuple):
        image.crop(top_left=(tup[1] , tup[0]), bottom_right=(tup[3], tup[2]), resize=(300,300))
        if image.img.ndim == 2:
          image.img = fill_nan(image.img, mode = 'linear')
        image.normalise()  #for [-1,1]
        #transpose the imgae only when there are three channels in the image
        if len(image.shape) == 3:
            image.img = image.img.transpose((2, 0, 1))
        #image_copy = np.copy(image.img)
        return image
    
    def make_tensors(self, fin_rgb, fin_depth):
        tensor = self.numpy_to_torch(np.concatenate(
        (np.expand_dims(fin_depth,0),fin_rgb), 0
        ))
        tensor = tensor.unsqueeze(0)  #make the tensor 4d
        return tensor

    @staticmethod
    def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))
        
    def show_crops(self):
        
        rows = len(self.coordinates.index)
        fig, ax = plt.subplots(rows, 2, figsize=(10, 5))
        if rows>0:
            for i, (obj, data) in enumerate(self.image_dict.items()):
                rgb_image = data['rgb_object'].transpose(1, 2, 0)  # Transpose back to HWC format for display
                depth_image = data['depth_object']
                if rows ==1:
                    try:
                        ax = np.expand_dims(ax, axis=0)
                    except Exception as e:
                        print(e) 
                ax[i,0].imshow(rgb_image)
                ax[i,0].set_title(f"{obj}_{self.coordinates.loc[i,'name']}")
                ax[i,0].axis('off')

                ax[i,1].imshow(depth_image, cmap = 'gray')
                ax[i,1].set_title(f"{obj}_{self.coordinates.loc[i,'name']}")
                ax[i,1].axis('off')
            plt.tight_layout()
            plt.show()
        pass
    
    def append_crops(self):
        
        pass 

def pred_grasps_and_display(model, image_dict, display_images:bool = False):
    
    for i, (obj, itype) in enumerate(image_dict.items()):
        with torch.no_grad():
            model_output = model(itype['tensor'])
        q_img, ang_img, width_img = post_process_output(model_output[0], model_output[1],
                                                    model_output[2], model_output[3])
        image_dict[obj].update([('q_img',q_img),('ang_img',ang_img),('width_img',width_img)])
        if display_images: # returns the grasps for only one object, make it return for multiple objects
            grasps = plot_output(rgb_img= itype['rgb_object'].img.transpose(1,2,0), depth_img= itype['depth_object'], 
                grasp_q_img= q_img, grasp_angle_img=ang_img, grasp_width_img=width_img, no_grasps=3, return_grasps= True)
        else:
            grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=3)

    
    return image_dict, grasps


def fill_nan(image, mode):
    if mode == 'zero':
        filled_image = image.copy()
        filled_image[np.isnan(filled_image) | ~np.isfinite(filled_image)] = 0
        return filled_image
    
    if mode == 'linear':
        x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        mask = np.isfinite(image)
        return interpolate.griddata((x[mask], y[mask]),image[mask],
                (x, y),
                method='linear')
    if mode == 'mean':
      nan_mask = np.isnan(image) | ~np.isfinite(image)
      filled_image = image.copy()
      filled_image[nan_mask] = 0  # Replace NaNs temporarily
      mean_filter = generic_filter(filled_image, np.nanmean, size=10, mode='constant')
      filled_image[nan_mask] = mean_filter[nan_mask]
      return filled_image
    if mode=='gaussian':
        nan_mask = np.isnan(image) | ~np.isfinite(image)
        filled_image = image.copy()
        filled_image[nan_mask] = 0
        blurred_image = gaussian_filter(filled_image, sigma=2)
        filled_image[nan_mask] = blurred_image[nan_mask]
        return filled_image


def linear_interpolation(image):
    x, y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
    mask = np.isfinite(image)
    return interpolate.griddata(
        (x[mask], y[mask]),
        image[mask],
        (x, y),
        method='linear'
    )

def fill_nan_with_mean(image):
    nan_mask = np.isnan(image) | ~np.isfinite(image)
    filled_image = image.copy()
    filled_image[nan_mask] = 0  # Replace NaNs temporarily
    mean_filter = generic_filter(filled_image, np.nanmean, size=10, mode='constant')
    filled_image[nan_mask] = mean_filter[nan_mask]
    return filled_image

if __name__ == '__main__':

    model = load_grasping_model()
