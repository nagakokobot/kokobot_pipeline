#import pyzed.sl as sl
import torch
from ultralytics import YOLO
#from datetime import datetime
import pandas as pd
#from PIL import Image
import numpy as np
from typing import Union
import matplotlib
#from yolov5.utils.plots import output_to_target

from helpers import get_model_path, sort_args, create_folder
from defaults import det_args


class Detector:
    '''
    This is a class returns the specific detector for object detection.
    - Gets the model from the default path.
    - Checks the saved models have architecture in them
    - TODO: How to handle video format.
    '''
    def __init__(self, detector_version: str = 'v5', device:str='cpu'):

        self.det_ver = detector_version
        self.device = device
        if self.device == 'cuda':
            self.device = torch.device(0 if torch.cuda.is_available() else 'cpu')
            print(f'Using device: {self.device}')
            
        #get the path where the model is saved.
        self.model_path = get_model_path(task = 'detection', version = self.det_ver)
        #self.model = self.check_architecture(model_path)

    def __enter__(self):
        self._current_backend = matplotlib.get_backend()
        self.model = self.check_architecture(self.model_path)
        return self

    def check_architecture(self, model_path):
        '''
        Check if the .pt file has the model architecture saved. else the architecture has to be loaded manually for inference
        - if no model architecture is saved in the saved model, model architecture has to be defined somewhere.
        '''

        if self.det_ver == 'v5':
            with torch.no_grad():
                model = torch.hub.load('ultralytics/yolov5', 'custom', path= model_path, device =self.device,_verbose=False, force_reload= True)
        else:
            model = YOLO(model_path, task='detect').to(self.device)

        print(f'Sucessfully loaded model from {model_path}')

        return model
    
    def del_model(self):
        if next(self.model.parameters()).is_cuda:
            print('Clearing gpu memory')
            torch.cuda.empty_cache()
        del self.model

    def __exit__(self, exc_type, exc_value, traceback):
        matplotlib.use(self._current_backend)
        

class Inference:
    '''
    TODO: 
    - also to show the inferences on the image for clear understanding
    '''

    def __init__(self, model, image:Union[str, np.ndarray], detector_version: str = 'v5', args :dict = None, project:str = None):
        self.model = model
        self.image = image
        self.det_ver = detector_version
        self.save_path = project
       # _, self.save_path = create_folder('inference_result',project)
        self.conf, self.iou, _ = process_args(args)
        self.res = self.get_inference()
        #pr = self.process_results(self.res)

    def get_inference(self):
        if self.det_ver == 'v5':
            self.model.conf = self.conf
            self.model.iou = self.iou
            self.model.multi_label = False
            results = self.model(self.image)
            results.save(save_dir = self.save_path+'/'+'inference_result')
        elif self.det_ver == 'v8':
            results = self.model.predict(self.image,conf = self.conf, iou = self.iou, save = True, project = self.save_path, name = 'inference_result')
        return results
    
    def process_results(self):
        '''
        In progress
        returns the detection results in s specific format
        '''
        if self.det_ver == 'v5':
            df = self.res.pandas().xyxy[0]
        elif self.det_ver == 'v8':
            #columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name']
            class_names = self.model.names

            all_boxes = []
            all_confidences = []
            all_class_ids = []
            all_class_names = []
            boxes = self.res[0].boxes
            for box in boxes:
                xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                cls_id = box.cls[0].item()  # Extract class ID
                conf = round(box.conf[0].item(), 2)  # Extract confidence score

                all_boxes.append([xmin, ymin, xmax, ymax])
                all_confidences.append(conf)
                all_class_ids.append(cls_id)
                all_class_names.append(class_names[cls_id])
            df = pd.DataFrame(all_boxes, columns=['xmin', 'ymin', 'xmax', 'ymax'])
            df['confidence'] = all_confidences
            df['class'] = all_class_ids
            df['name'] = all_class_names
        df.to_csv(self.save_path+'/inference_result/detections.csv')
        return df

    
def process_args(args:dict):
    '''
    get the default detection args and compare it with the user defined args and returns user defirned args.
    '''
    options = det_args()
    conf = options['det_conf']
    iou = options['iou']
    img_sz = options['img_sz']
    possible_args = sort_args(args, options)
    if 'det_conf' in possible_args:
        conf = args['det_conf']
    if 'iou' in possible_args:
        iou = args['iou']
    return conf, iou, img_sz

        

if __name__ == '__main__':

    #detect
    image_path_720 = '/home/student/naga/kokobot_pipeline/test_folder/rgb_image_720.png'
    image_path_1080 = '/home/student/naga/kokobot_pipeline/test_folder/rgb_image_1080.png'
    model_ver = 'v8'
    device = 'cuda'
    
    args = {'det_conf': 0.85,
            'iou': 0.7}
    
    detector = Detector(detector_version=model_ver, device = device)
    
    res = Inference(model = detector.model, image= image_path_1080, detector_version= model_ver, args = args)
    pr = res.process_results()
    print(pr)
    #res.show()
    res2 = Inference(model = detector.model, image= image_path_720, detector_version= model_ver, args = args)
    pr2= res.process_results()
    print(pr2)
    detector.del_model()

    #np_output = output_to_target(res.res[0])

    



