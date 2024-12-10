import os
import torch
from typing import Union
import numpy as np
import pandas as pd
from imageio.v2 import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from ggcnn.models.ggcnn2 import GGCNN2

from PIL import Image
#for filling the nan and inf values
from scipy import interpolate
from scipy.ndimage import generic_filter, gaussian_filter

from helpers import get_model_path
from segment import *
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
    def __init__(self,rgb:Union[str, np.ndarray], depth:Union[str, np.ndarray], coordinates: Union[str, pd.DataFrame], include_segmentation:bool= True):
        self.rgb = rgb
        self.depth = depth
        self.coordinates = coordinates
        self.include_segmentation = include_segmentation
        if self.include_segmentation:
            self.seg_model = Segment()
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
                    
                    image_dict[obj] = {}
                    for i, crop_object in enumerate([rgb_i, depth_i]):
                      crop_object.crop(top_left=(ymin, xmin), bottom_right=(ymax,xmax), resize=(300,300))
                      if i == 0:
                        image_dict[obj].update([('org_rgb', rgb_i.img.copy())])
                      else:
                        image_dict[obj].update([('org_depth', depth_i.img.copy())])
                      segmentation_result = self.process_image(image = crop_object, tup = (xmin, ymin, xmax, ymax))
                      if segmentation_result:
                          image_dict[obj].update([('seg_res', segmentation_result)])
                    #self.process_image(image = depth_i, tup = (xmin, ymin, xmax, ymax), segmentation= self.include_segmentation)
                    image_dict[obj].update([('formatted_crop_rgb', rgb_i),('formatted_crop_depth', depth_i)])
                    tensor_i = self.make_tensors(fin_rgb=rgb_i.img, fin_depth= depth_i.img)
                    image_dict[obj].update([('tensor', tensor_i)])
                    self.temp_res = None
                    del rgb_i, depth_i
        #print(no_of_objects)
        return image_dict
    

    def process_image(self, image, tup:tuple, segmentation:bool = True):
        #image.crop(top_left=(tup[1] , tup[0]), bottom_right=(tup[3], tup[2]), resize=(300,300))
        seg_res = None
        if len(image.shape) == 3 and self.include_segmentation:
            seg_res = self.seg_model.predict(image.img.copy())
        if image.img.ndim == 2:
          image.img = fill_nan(image.img, mode = 'linear')
          '''
          if self.include_segmentation:
            masks = overlap_masks(self.temp_res, image.img)
            if 'metal_part' in masks.keys():
              image.img =image.img* masks['metal_part']
              print(np.max(image.img), np.min(image.img))
              segmentation_mask = ''  # two arrays with metal and plastic part later apply to the original image
          '''
        image.normalise()  #for [-1,1]
        #transpose the imgae only when there are three channels in the image
        if len(image.shape) == 3:
            image.img = image.img.transpose((2, 0, 1))
        #image_copy = np.copy(image.img)
        if self.include_segmentation:
            return seg_res
        else:
          return None
    
    def make_tensors(self, fin_rgb, fin_depth):
        print('supposed dimensions of fin_rgb are:', fin_rgb.shape)
        print('supposed dimensions of fin_depth are:', fin_depth.shape)
        tensor = self.numpy_to_torch(np.concatenate(
        (np.expand_dims(fin_depth,0),fin_rgb), 0
        ))
        tensor = tensor.unsqueeze(0)  #make the tensor 4d
        print(tensor.shape)
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
                rgb_image = data['formatted_crop_rgb'].transpose(1, 2, 0)  # Transpose back to HWC format for display
                depth_image = data['formatted_crop_depth']
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
            grasps = plot_output(rgb_img= itype['formatted_crop_rgb'].transpose(1,2,0), depth_img= itype['formatted_crop_depth'], 
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


  image_path = 'project_aux/2024-12-10_14-46-57/rgb_image.png'
  depth_path = 'project_aux/2024-12-10_14-46-57/true_depth.tiff'
  coordinates = 'project_aux/2024-12-10_14-46-57/inference_result/detections.csv'


  model = load_grasping_model()
  crops = Process_crops(rgb = image_path, depth = depth_path, coordinates= coordinates, include_segmentation= True)
  img_dict = crops.image_dict
  hammer_dict = img_dict['object_1_Hammer']
  print(hammer_dict.keys())
  #crops.show_crops()

  seg_res = hammer_dict['seg_res']
  for res in seg_res:
      cls_names = res.names
      for i, m in enumerate(res):
        m_i= m.masks.data.cpu().numpy()
        label = cls_names[m.boxes.cls.tolist().pop()]
        m_i_resize = resize(np.squeeze(m_i.transpose(1,2,0)), (300,300), preserve_range= True).astype(hammer_dict['formatted_crop_rgb'].img.dtype)
        temp_img = hammer_dict['formatted_crop_depth'] * m_i_resize
        if label in hammer_dict.keys():
          label = label+str(i)
        hammer_dict.update([(label,temp_img)])
  print(hammer_dict.keys())
  which_seg = 'metal_part2'
  new_tensor = crops.make_tensors(fin_rgb=hammer_dict['formatted_crop_rgb'].img, fin_depth=hammer_dict[which_seg])
  #plt.imshow(hammer_dict['metal_part'])
  #plt.show()
  use_seg = True
  if use_seg:
    for i, (obj, itype) in enumerate(img_dict.items()):
      with torch.no_grad():
          model_output = model(new_tensor)
      q_img, ang_img, width_img = post_process_output(model_output[0], model_output[1],
                                                  model_output[2], model_output[3])
      grasps = plot_output(rgb_img=itype['formatted_crop_rgb'].transpose(1,2,0), depth_img= itype[which_seg], 
              grasp_q_img= q_img, grasp_angle_img=ang_img, grasp_width_img=width_img, no_grasps=3, return_grasps= True)
      grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=3)
  else:
    image_dict, grasps = pred_grasps_and_display(model = model, image_dict= crops.image_dict, display_images= True)
  