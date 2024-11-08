# kokobot_pipeline
Contains the code for the thesis 'Grasp detection and tool handover by cobots'.

## Project structure
```bash
Kokobot_pipeline/ 
├── pre_trained_models
│   ├── detection                           
│   │   ├── v5
│   │   │   ├── best.pt
│   │   │   └── last.pt
│   │   └── v8
│   │       ├── best.pt
│   │       └── last.pt
│   └── grap_synthesis
├── project_aux                             #contains folders required for runtime
│   └── 2024-10-15_13-58-29                 # created directory in the runtime to work on (structure could be changes as the project goes on)
│       ├── initParameters.conf.yml         # initparameters for the zed camera
│       ├── rgb_image.png                   # rgb image
│       ├── runtimeParameters.yml           #  # the runtime parameters for the zed camera
│       └── true_depth.tiff                 #depth image
├── README.md                               # Documentation for your project
└── src                                     # Source code folder 
    ├── camera_parameters.py                # gets the parameters for the camera
    ├── camera.py                           # get the images in rgb and depth format
    ├── defaults.py                         # contains the default values and possible options for running the scripts
    ├── detection.py                        # carries out detection, generating rtx engines for saved models
    ├── helpers.py                          # contains all the helper functions for above scripts
    └── main.py                             # entry point to the code
```
## Description
- This project is to enable the cobots to precive the work space to identify 8 classes of industrial tools on the working table of the robot and generate a stable grasp coordinates to pick the tool.
- This goal is acheived in several steps:
 - [`Object detection`](#object-detection) --*perception*-->`Image Transformations`-->[`Grasp Synthesis`](#grasp-synthesis)--*Stable Grasps*-->`Coordinate transformation`-->`Coordinates for the robot arm`.   


## Object detection
- The pipeline is compatible with yolov5 and yolov8. Train the models on the custom data and save it in the appropriate directory as in in above [tree structure](#project-structure).
- Follow the official Ultralytics page **https://docs.ultralytics.com/models/** to load, train and export the model. 

## Grasp Synthesis
- The grasp synthesis is based on the work `Closing the Loop for Robotic Grasping: A Real-time, Generative Grasp Synthesis Approach`,[arXiv](https://arxiv.org/abs/1804.05172)
- Also refer to the original implementation in pytorch: [repo](https://github.com/dougsm/ggcnn/tree/master)
- In the present work [Jacquard dataset](https://jacquard.liris.cnrs.fr/) is choosen as it is bigger than coronell dataste and has a wide variety of objects with various perspectives