import pyzed.sl as sl
import torch, torchvision

from helpers import get_model_path


class detection:
    '''
    This is a class returns the specific detector for object detection.
    - Gets the model from the default path.
    - Checks the saved models have architecture in them
    - To generate the rtx engine if needed
    - 
    '''
    def __init__(self, workfolder: str, detector_version: str = 'v5'):

        self.folder = workfolder
        self.det_ver = detector_version

        #get the path where the model is saved.
        model_path = get_model_path(task = 'detection', version = self.det)

    def check_architecture(self,model_path: str):
        '''
        Check if the .pt file has the model architecture saved. else the architecture has to be loaded manually for inference
        - if no model architecture is saved in the saved model, model architecture has to be defined somewhere.
        '''
        model = torch.load(model_path)
        #check for architecture
        if isinstance(model, torch.nn.Module):
            return True, model
        else:
            raise TypeError(f'The model at the path {model_path}, has no model architecture saved.')

