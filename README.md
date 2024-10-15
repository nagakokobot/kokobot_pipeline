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