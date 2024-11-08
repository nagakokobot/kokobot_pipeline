#import sys, os

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ggcnn.models.ggcnn2 import GGCNN2


model = GGCNN2(input_channels=4)
print(model)
