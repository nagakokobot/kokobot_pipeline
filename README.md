# kokobot_pipeline
Contains the code for the thesis 'Grasp detection and tool handover by cobots'.

## Project structure
'''bash
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
├── project_aux
│   └── 2024-10-15_13-58-29
│       ├── initParameters.conf.yml
│       ├── rgb_image.png
│       ├── runtimeParameters.yml
│       └── true_depth.tiff
├── README.md
└── src
    ├── camera_parameters.py
    ├── camera.py
    ├── defaults.py
    ├── detection.py
    ├── helpers.py
    └── main.py
'''