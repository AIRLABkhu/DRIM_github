# DRIM: Depth Restoration with Interference Mitigation in Multiple LiDAR Depth Cameras

## 0. Summary

### Abstract
```
LiDAR depth cameras are widely used for accurate depth measurements in numerous studies.
However, a drawback of these cameras is that interference occurs in multiple camera systems, resulting in artifacts in the depth data.
These artifacts pose challenges for restoration using existing image restoration methods.
In this paper, we propose a novel approach, DRIM for depth restoration.
Our method begins to distinguish between artifacts of interfered depth.
We then propose a method that utilizes a shared encoder structure to predict these artifacts, leveraging them to restore depth.
Previously, no dataset was available for learning interference in multiple LiDAR depth cameras.
Therefore, we create and provide a depth interference dataset for the first time.
Our experiments demonstrate superior depth restoration performance compared to other image restoration methods, and we show the capability to restore depth in challenging scenarios. 
Through ablation studies, we confirm that classifying and utilizing artifacts is efficient for depth restoration.
Also, we demonstrate the effectiveness of training by employing a shared encoder structure.
```
### Framework
<img width="1080" alt="main_architecture_3d" src="https://github.com/user-attachments/assets/308e0f7e-2ffb-49ce-9341-ddded2ae2cda">

## 1. Usage

### 1.1 Setup

- Conda environment
```
conda env create -f environment.yaml
```

### 1.2 Training

Follow the instructions below to begin traning our model.

```
python3 train.py --exp 'save/path/to/your/folder'
```

### 1.3 Testing

Follow the instructions below to begin testing our model.

The best weights are [here](https://drive.google.com/drive/folders/1ANEa7L_j5Oz2kwvDbXlFHDLBR0aHwXF1?usp=drive_link).
```
python3 test.py --exp 'weights/path/to/your/folder'
```

Follow the instructions below to begin testing our model in different scenarios.
```
python3 test_other_l515.py --exp 'weights/path/to/your/folder'
python3 test_other_kinect.py --exp 'weights/path/to/your/folder'
```

## 2. License

The code is released under the MPL 2.0 License. MPL is a copyleft license that is easy to comply with. You must make the source code for any of your changes available under MPL, but you can combine the MPL software with proprietary code, as long as you keep the MPL code in separate files.


## 3. Code reference:

Our code is based on the following repositories:

- [saic_depth_completion](https://github.com/SamsungLabs/saic_depth_completion/tree/master)
