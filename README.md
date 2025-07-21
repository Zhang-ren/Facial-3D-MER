# 3D Facial Micro-Expression Recognition System

**Official implementation** of the paper: [Facial 3D Regional Structural Motion Representation Using Lightweight Point Cloud Networks for Micro-Expression Recognition](Facial_3D_Regional_Structural_Motion_Representation_Using_Lightweight_Point_Cloud_Networks_for_Micro-Expression_Recognition.pdf)

> Ren Zhang, Jianqin Yin, Chao Qi, Yonghao Dang, Zehao Wang, Zhicheng Zhang, and Huaping Liu. Facial 3D Regional Structural Motion Representation Using Lightweight Point Cloud Networks for Micro-Expression Recognition. IEEE Transactions on Affective Computing, pages 1–15, 2025.

A facial micro-expression recognition system based on 3D point clouds and optical flow, using deep learning methods for expression classification.

## Project Overview

This project implements a complete facial micro-expression recognition pipeline, including:
- Face detection and cropping
- Optical flow computation
- 3D point cloud generation and processing
- Point cloud-based deep learning classification

## Data Preparation

Before training the model, you need to preprocess the original data:

### 1. Data Preprocessing

Data preprocessing consists of three steps, all scripts are located in the `preprocessing` folder:

```bash
cd preprocessing

# Step 1: Face detection and cropping
python step1_crop_face.py

# Step 2: Optical flow computation
python step2_opticalflow.py

# Step 3: Point cloud processing
python step3_pointcloud.py
```

> **Note**: The preprocessing scripts use the `config.yaml` file for configuration. Please modify this configuration file according to your data paths.

### 2. External Dependencies

#### Optical Flow Extraction
This project uses the optical flow algorithm from [Ce Liu's Optical Flow](https://people.csail.mit.edu/celiu/OpticalFlow/). Please download the package from the website and extract it to the `preprocessing` directory.

The optical flow alignment is implemented based on the method described in the paper ["A Main Directional Mean Optical Flow Feature for Spontaneous Micro-Expression Recognition"](https://ieeexplore.ieee.org/document/7286757), which has been implemented in `affine.m`.

#### Neural Network Framework
The network code is based on [Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). Please download the repository and extract it to the project directory. Do not overwrite existing files that have been modified for this project.

### 3. Configuration File

The `config.yaml` file contains all path and parameter settings, including:
- Data path configuration
- File path configuration
- Folder structure
- Expression type definitions
- Emotion label mappings
- Point cloud processing parameters

## Environment Setup

### 1. Install Dependencies

Use the following command to install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. MATLAB Engine

Some optical flow calculations require MATLAB engine support. Please ensure that MATLAB is installed and the Python interface is configured.

## Model Training

### Running Training

Use the provided script to run training:

```bash
./run.sh [experiment_name] [batch_size] [GPU_ID]
```

Parameter description:
- `experiment_name`: Name of the training experiment, used to save models and logs
- `batch_size`: Training batch size
- `GPU_ID`: Specify the GPU number to use (e.g., 0, 1, 2)

Example:

```bash
./run.sh exp7_classification 32 0
```

### Training Options

You can also run the training script directly and specify more parameters:

```bash
python train_classification.py --exp_name exp7 --batch_size 32 --gpu 0
```

## Model Testing

After training is complete, you can use the test script to evaluate model performance:

```bash
python test_classification.py --exp_name [experiment_name] --model_path [model_path]
```

## Project Structure

```
.
├── config.yaml              # Configuration file
├── preprocessing/           # Data preprocessing scripts
│   ├── step1_crop_face.py   # Face detection and cropping
│   ├── step2_opticalflow.py # Optical flow computation
│   └── step3_pointcloud.py  # Point cloud processing
├── models/                  # Model definitions
├── utils/                   # Utility functions
├── data_utils/              # Data loading and processing
├── train_classification.py  # Classification model training script
├── test_classification.py   # Classification model testing script
├── run.sh                   # Training run script
└── requirements.txt         # Dependency list
```

## Citation

If you use this project in your research, please cite our paper:

```
@article{zhang2025facial,
  title={Facial 3D Regional Structural Motion Representation Using Lightweight Point Cloud Networks for Micro-Expression Recognition},
  author={Zhang, Ren and Yin, Jianqin and Qi, Chao and Dang, Yonghao and Wang, Zehao and Zhang, Zhicheng and Liu, Huaping},
  journal={IEEE Transactions on Affective Computing},
  pages={1--15},
  year={2025},
  publisher={IEEE}
}
```

Also, please cite the following works that this project builds upon:

```
@article{liu2009beyond,
  title={Beyond pixels: exploring new representations and applications for motion analysis},
  author={Liu, Ce},
  journal={Doctoral Thesis, Massachusetts Institute of Technology},
  year={2009}
}

@article{xu2015main,
  title={A Main Directional Mean Optical Flow Feature for Spontaneous Micro-Expression Recognition},
  author={Xu, Feng and Zhang, Junping and Wang, James Z},
  journal={IEEE Transactions on Affective Computing},
  volume={9},
  number={1},
  pages={50--62},
  year={2015},
  publisher={IEEE}
}

@article{Pytorch_Pointnet_Pointnet2,
  author={Xu Yan},
  title={Pointnet/Pointnet++ Pytorch},
  journal={https://github.com/yanx27/Pointnet_Pointnet2_pytorch},
  year={2019}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
