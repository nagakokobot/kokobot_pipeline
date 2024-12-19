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
from segment import Segment
from transform import transform_grasps
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
    def __init__(self,rgb:Union[str, np.ndarray]=None, depth:Union[str, np.ndarray]=None, coordinates: Union[str, pd.DataFrame]=None, include_segmentation:bool= True):
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
        if not self.rgb is None:
          self.image_dict = self.get_image_crops()
        else:
            pass
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
                    obj_pos_dict = {'aspect_x': (xmax-xmin)/300,'aspect_y': (ymax-ymin)/300, 'xmin':xmin, 'ymin': ymin}
                    image_dict[obj].update([('object_position',obj_pos_dict)])
                    for i, crop_object in enumerate([rgb_i, depth_i]):
                      crop_object.crop(top_left=(ymin, xmin), bottom_right=(ymax,xmax), resize=(300,300))
                      #if i == 0:
                        #image_dict[obj].update([('org_rgb', rgb_i.img.copy())])
                     # else:
                        #image_dict[obj].update([('org_depth', depth_i.img.copy())])
                      segmentation_result = self.process_image(image = crop_object)
                      if segmentation_result:
                          image_dict[obj].update([('seg_res', segmentation_result)])
                    #self.process_image(image = depth_i, tup = (xmin, ymin, xmax, ymax), segmentation= self.include_segmentation)
                    image_dict[obj].update([('formatted_crop_rgb', rgb_i),('formatted_crop_depth', depth_i)])
                    tensor_i = self.make_tensors(fin_rgb=rgb_i.img, fin_depth= depth_i.img)
                    image_dict[obj].update([('tensor', tensor_i)])
                    del rgb_i, depth_i
        #print(no_of_objects)
        return image_dict
    

    def process_image(self, image):
        #image.crop(top_left=(tup[1] , tup[0]), bottom_right=(tup[3], tup[2]), resize=(300,300))
        seg_res = None
        if len(image.shape) == 3 and self.include_segmentation:
            seg_res = self.seg_model.predict(image = image.img.copy())
        if image.img.ndim == 2:
          image.img = fill_nan(image.img, mode = 'linear')
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
        '''
        fin_rgb.shape: (3,300,300)
        fin_depth.shape : (300,300)
        output- tensor.shape: ([1,4, 300,300])
        '''
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
                rgb_image = data['formatted_crop_rgb'].transpose(1, 2, 0)  # Transpose back to HWC format for display
                depth_image = data['formatted_crop_depth']
                if rows ==1:
                    try:
                        ax = np.expand_dims(ax, axis=0)
                    except Exception as e:
                        print(e) 
                ax[i,0].imshow(rgb_image)
                ax[i,0].set_title(f"{obj}_rgb")
                ax[i,0].axis('off')

                ax[i,1].imshow(depth_image, cmap = 'gray')
                ax[i,1].set_title(f"{obj}_depth")
                ax[i,1].axis('off')
            plt.tight_layout()
            plt.show()
        pass
    

def pred_grasps_and_display(model, image_dict, display_images:bool = False):
    '''
    Works without segmentation enabled
    '''
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

def make_tensors_and_predict_grasps(image_dict, grasp_model):

    #grasps_dict = {}
    new_tens = Process_crops()
    for obj,itype in image_dict.items():
        itype['grasps'] = {}
        grasps_dict = {'pipeline_grasp':{}}
        obj_pos = itype['object_position']
        p_grasp = predict_grasp_on_tensor(model = grasp_model,tensor=itype['tensor'])
        temp_pimage_grasps = transform_grasps(grasps=p_grasp['object_grasps'], obj_loc = obj_pos)
        p_grasp.update([('image_grasps', temp_pimage_grasps)])
        grasps_dict.update([('pipeline_grasp',p_grasp)])
        seg_dict = itype['segmentation_masks']
        if seg_dict is not None:
          for depth_name, depth in seg_dict.items():
              grasps_dict[depth_name] = {}
              temp_tensor = new_tens.make_tensors(fin_rgb=itype['formatted_crop_rgb'], fin_depth=depth)
              temp_grasps = predict_grasp_on_tensor(model=grasp_model, tensor = temp_tensor)
              temp_image_grasps = transform_grasps(grasps=temp_grasps['object_grasps'], obj_loc = obj_pos)
              temp_grasps.update([('image_grasps', temp_image_grasps)])
              grasps_dict.update([(depth_name,temp_grasps)])
              
        else:
            print(f'The segmentation masks for {obj} are empty')
        itype.update([('grasps',grasps_dict)])
    return image_dict

def predict_grasp_on_tensor(model, tensor):
    with torch.no_grad():
        op = model(tensor)
    q_img, ang_img, width_img = post_process_output(op[0], op[1],
                                                    op[2], op[3])
    grasps = detect_grasps(q_img, ang_img, width_img=width_img, no_grasps=3)
    grasps = {'q_img':q_img, 'ang_img':ang_img,
                    'width_img':width_img, 'object_grasps':grasps}


    return grasps


def display_grasps_per_object(image_dict):
    for obj, itype in image_dict.items():
        obj_grasp_fig = plt.figure(figsize=(10,10))
        obj_grasp_fig.suptitle(obj)
        grasps_dict = itype['grasps']
        pos1 = 0
        for i, (depth_name, grasp) in enumerate(grasps_dict.items()):
          pos2  = pos1+i+1
          if depth_name == 'pipeline_grasp':
            depth_img = itype['formatted_crop_depth']
          else:
            depth_img = itype['segmentation_masks'][depth_name]
          ax = obj_grasp_fig.add_subplot(len(grasps_dict), 2,pos2)
          ax.imshow(itype['seg_res'][0].orig_img)
          ax.set_title('object_grasps')
          for g in grasp['object_grasps']:
            g.plot(ax)
          ax2 = obj_grasp_fig.add_subplot(len(grasps_dict),2,pos2+1)
          ax2.imshow(depth_img)
          ax2.set_title(depth_name)
          pos1+=1
    plt.show()
    pass

def display_grasps_per_image(image_dict, org_image):
    '''
    Displays the best grasps per object per type(pipeline/metsl/plastic)
    '''
    if isinstance(org_image, str):
        org_image = plt.imread(org_image)
    fig = plt.figure()
    ax1 = fig.add_subplot(1,1,1)
    ax1.imshow(org_image)
    for obj, itype in image_dict.items():
        grasps_dict = itype['grasps']
        for ty,se in grasps_dict.items():
            gr = se['image_grasps']
            if len(gr)>0:
              gr[0].plot(ax1)
    
    plt.show()

    pass

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

  from segment import get_masks_from_seg_res
  image_path = 'project_aux/test_folder1/rgb_image.png'
  depth_path = 'project_aux/test_folder1/true_depth.tiff'
  coordinates = 'project_aux/test_folder1/inference_result/detections.csv'


  model = load_grasping_model()
  crops = Process_crops(rgb = image_path, depth = depth_path, coordinates= coordinates, include_segmentation= True)
  img_dict = crops.image_dict
  img_dict = get_masks_from_seg_res(image_dict=img_dict)

  img_dict = make_tensors_and_predict_grasps(image_dict= img_dict,grasp_model=model)
  #obj_1_keys = img_dict[list(img_dict.keys())[0]].keys()
  #print(img_dict[list(img_dict.keys())[0]]['object_grasps'].keys())
  display_grasps_per_object(img_dict)
  display_grasps_per_image(image_dict=img_dict, org_image=image_path)
  '''
  gr = img_dict['object_1_Scissor']['grasps']['plastic_part']
  ob_gr = gr['object_grasps']
  im_gr = gr['imageitype_grasps']
  print('object_grasps values for scissor_plastic_part: ', 'center - ', ob_gr[0].center, 'width - ',ob_gr[0].width, 'length - ', ob_gr[0].width)
  print('image_grasps values for scissor_plastic_part: ', 'center - ', im_gr[0].center, 'width - ',im_gr[0].width, 'length - ', im_gr[0].width)
  
  fig = plt.figure()
  ax1 = fig.add_subplot(1,2,1)
  ax1.imshow(plt.imread(image_path))
  for g_i in im_gr:
      ax1.plot(g_i.center[1], g_i.center[0], 'x')
      g_i.plot(ax1)
  ax2 = fig.add_subplot(1,2,2)
  ax2.imshow(img_dict['object_1_Scissor']['formatted_crop_rgb'].transpose(1, 2, 0))
  for g in ob_gr:
      ax2.plot(g.center[1], g.center[0], 'x')
      g.plot(ax2)
  plt.show()
  '''
