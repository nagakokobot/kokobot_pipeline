import pyzed.sl as sl
import torch, torchvision
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
from PIL import Image

from helpers import get_model_path


class Detector:
    '''
    This is a class returns the specific detector for object detection.
    - Gets the model from the default path.
    - Checks the saved models have architecture in them
    - TODO: To generate the rtx engine if needed
    - TODO: Does the class need a workfolder or will it handle it in the main.py?
    - TODO: Also make the class set the model parameters from the user.
    '''
    def __init__(self, workfolder: str = None, detector_version: str = 'v5'):

        self.folder = workfolder
        self.det_ver = detector_version

        #get the path where the model is saved.
        self.model_path = get_model_path(task = 'detection', version = self.det_ver)
        print(self.model_path)
        self.model = self.check_architecture()


    def check_architecture(self):
        '''
        Check if the .pt file has the model architecture saved. else the architecture has to be loaded manually for inference
        - if no model architecture is saved in the saved model, model architecture has to be defined somewhere.
        '''

        if self.det_ver == 'v5':
            #check for achitecture
            model = torch.load(self.model_path)
            if isinstance(model, torch.nn.Module):
                pass
            else:
                model = torch.hub.load('ultralytics/yolov5', 'custom', path= self.model_path)
        else:
            model = YOLO(self.model_path, task='detect')

        print(f'Sucessfully loaded model from {self.model_path}')

        return model


class Inference:
    '''
    TODO: 
    -to handle image as a path or sl.mat() [Preffered, this could eliminate the need to load the image again from system path]. This enables video frames also could be detected for objects.
    - also to show the inferences on the image for clear understanding
    '''

    def __init__(self, model, image_path:str,device:str='cpu', detector_version: str = 'v5'):
        self.model = model
        self.image = image_path
        self.device = device
        self.det_ver = detector_version
        if self.device == 'cuda':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            print('Found cuda, moving model to gpu for inference')
            self.model.to(self.device)
            #self.image = self.preprocess_image()
        else:
            print('Using cpu for Inference')

        res = self.get_inference()
        pr = self.process_results(res)


    def get_inference(self):
        if self.det_ver == 'v5':
            results = self.model(self.image)
        elif self.det_ver == 'v8':
            results = self.model.predict(self.image)
        return results
    
    def process_results(self, results):
        '''
        In progress
        returns the detection results in s specific format
        '''
        if self.det_ver == 'v5':
            df = results.pandas().xyxy[0]
        elif self.det_ver == 'v8':
            columns = ['xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name']
            df = pd.DataFrame(columns=columns)
            
        return df

    def preprocess_image(self):

        '''
        TODO: Check if this class is even necessary
        '''
        image = Image.open(self.image).convert('RGB') 
        preprocess = torchvision.transforms.Compose([
            torchvision. transforms.Resize((640, 640)),  # Resize to the model's expected input size
            torchvision. transforms.ToTensor(),  # Convert the image to a tensor
            ])
        image_tensor = preprocess(image).unsqueeze(0)  # Add a batch dimension
        
        return image_tensor
    





def run_inference(model,image_path:str, device:str='cpu'): # changed to a class later on
    '''TODO: 
    -to handle image as a path or sl.mat() [Preffered, this could eliminate the need to load the image again from system path]. This enables video frames also could be detected for objects.
    - also to show the inferences on the image for clear understanding
    '''    
    #empty dataframe:
    #result = pd.DataFrame(columns = ['x_min', 'ymin', 'x_max', 'y_max', 'class'])
    # Set the device
    if device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using {device} for detection')
        model.to(device)
        # Now pass the images to gpu for gpu processing
        image = Image.open(image_path).convert('RGB') 
        preprocess = torchvision.transforms.Compose([
            torchvision. transforms.Resize((640, 640)),  # Resize to the model's expected input size
            torchvision. transforms.ToTensor(),  # Convert the image to a tensor
            ])
        image_tensor = preprocess(image).unsqueeze(0)  # Add a batch dimension
        det = model(image_tensor)
        print('*'*10,'using gpu','*'*10)
        print(det)
    elif device == 'cpu':
        print('*'*10,'using cpu','*'*10)
        det = model(image_path)
        print(det)    
    
    # Process and return results
    
    return det
        

if __name__ == '__main__':

    #detect
    image_path_720 = '/home/student/naga/kokobot_pipeline/test_folder/rgb_image_720.png'
    image_path_1080 = '/home/student/naga/kokobot_pipeline/test_folder/rgb_image_1080.png'

    
    #yolov8
    # Run inference on GPU
    #res1 =run_inference(model = detector_v5.model, image_path = image_path_720, device='cuda')
    # Run inference on CPU
    #res2 = run_inference(model = detector_v5.model, image_path = image_path_720)

    #inf = Inference(model = detector_v5.model, image_path = image_path_720)#, device='cuda')
    #print(inf.detections.pandas().xyxy[0])
    #inf.detections[0].show
    #for det in inf.detections:
        #x1, y1, x2, y2, confidence, cls = det.tolist()
        