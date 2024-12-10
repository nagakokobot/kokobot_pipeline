from ultralytics import YOLO
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

from helpers import get_model_path



def load_seg_model(name:str= None):
  path = get_model_path(task = 'segmentation',model_name= name)
  print(path)
  model = YOLO(path)
  
  pass


class Segment:

  def __init__(self, model_name:str = None):
    self.model_name = model_name
    self.model_path = get_model_path(task = 'segmentation', model_name=self.model_name)
    self.model = YOLO(self.model_path)
    
  def predict(self, image:Union[str, np.ndarray]):
    if isinstance(image, str):
      image = plt.imread(image)
      if np.max(image) < 2:
        image = image*255
    result = self.model(image)
    #masks = result[0].masks.data.cpu().numpy()
    '''
    result_dict = {}
    for r in result:
      for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]
        temp_mask = c.masks.data.cpu().numpy()
        result_dict.update([label, temp_mask])
    '''
    return result
  

def overlap_masks(result, image):
  org_width, org_height = image.shape[:2]
  print('passed image shape is: ', image.shape)  #has shape (300,300)
  masks_dict = {}
  for r in result:
      for ci, c in enumerate(r):
        label = c.names[c.boxes.cls.tolist().pop()]
        temp_mask = c.masks.data.cpu().numpy().transpose(1,2,0)
        temp_mask = np.squeeze(temp_mask)
        print(f'temp_mask shape before reshapeing of object {label} is: ',temp_mask.shape)
        temp_mask = resize(temp_mask, (image.shape[0],image.shape[1]),preserve_range= True).astype(image.dtype)
        print(f'temp_mask shape after reshapeing of object {label} is: ',temp_mask.shape)
        if label in masks_dict.keys():
          label = label+'_'+str(ci)
        masks_dict.update([(label, temp_mask)])
 
  return masks_dict
    



if __name__ == '__main__':
  #load_seg_model()
  from grasp_synthesis import Process_crops
  image_path = 'project_aux/test_folder1/rgb_image.png'
  depth_path = 'project_aux/test_folder1/true_depth.tiff'
  coordinates = 'project_aux/test_folder1/inference_result/detections.csv'

  object_crops = Process_crops(rgb=image_path, depth=depth_path, coordinates=coordinates, include_segmentation = True)
  print(object_crops.image_dict)
  plt.imshow(object_crops.image_dict['object_1_Hammer']['rgb_object'].transpose(1, 2, 0))
  plt.show()
  seg_model = Segment()
  mask = seg_model.predict(image = object_crops.image_dict['object_1_Hammer']['rgb_object'].transpose(1, 2, 0))
  print(mask.shape)