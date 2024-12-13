from ultralytics import YOLO
from typing import Union
import numpy as np
import matplotlib.pyplot as plt
#from skimage.transform import resize

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

    result = self.model(image, conf = 0.8 , imgsz = 320, verbose = False, retina_masks=True)
    return result

    
def get_masks_from_seg_res(image_dict):

  cls_names = {0:'metal_part', 1:'plastic_part'}
  for obj, itype in image_dict.items():
      '''
      iterate through the image dic and produce three masks for each object crop inside
      consisting of metal_part, plastic_part and full image and (another function to make tensors and prediction, or update the exisitng function of 'pred_grasps_and_display')
      '''
      #itype['segmentation_masks'] = {'plastic_part':{}, 'metal_part':{}}
      itype['segmentation_masks'] = {}
      res = itype['seg_res'][0]
      cls = res.boxes.cls.tolist()
      masks = res.masks.data.cpu().numpy()
      if len(masks)>0:
        for i, m in enumerate(masks):
          #print(res[i].boxes.cls.tolist().pop())
          label = cls_names[int(cls[i])]
          #m_i_resize = resize(m, (300,300), preserve_range= True).astype(res.orig_img.dtype)
          temp = itype['formatted_crop_depth']*m
          if label in itype['segmentation_masks'].keys():
            label = label+str(i)
          itype['segmentation_masks'].update([(label, temp)])
      else:
        print(f'No segmentation mask result found for {obj}')
      
  return image_dict


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