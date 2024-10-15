# kokobot_pipeline
Contains the code for the thesis 'Grasp detection and tool handover by cobots'.

## Project structure
Kokobot_pipeline/ 
├── project_aux                                 #contains folders required for runtime 
│    ├── workspace                              # created directory in the runtime to work on (structure could be changes as the project goes on)
│           ├── initParameters.conf.yml         # initparameters for the zed camera 
│           ├── runtimeParameters.yml           # the runtime parameters for the zed camera
│           ├── images                          # all the necessary images of all the formats (rgb, depth..)
│
├── pre_trained_models                          # has the pretrained models for all the tasks 
│   ├── detection
│       ├── v5
│           ├── model.pt
│       └── v8
│           ├── model.pt
│   └── grasp_synthesis 
│
├── src/                                        # Source code folder 
│   ├── main.py                                 # entry point to the code
│   ├── camera_parameters.py                    # gets the parameters for the camera
│   ├── camera.py                               # get the images in rgb and depth format
│   ├── detection.py                            # carries out detection, generating rtx engines for saved models
│   ├── helpers.py                              # contains all the helper functions for above scripts
│   └── defaults.py                             # contains the default values and possible options for running the scripts
│ 
├── .gitignore                                  # Specifies files to ignore by Git 
├── README.md                                   # Documentation for your project 
├── requirements.txt                            # Python dependencies (used by pip)