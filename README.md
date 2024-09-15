# DRIM: Depth Restoration with Interference Mitigation in \\Multiple LiDAR Depth Cameras

## 0. Summary

### Abstract
```

```
### Framework


## 1. Usage

### 1.1 Setup

- Conda environment
```

```

### 1.2 Training

Follow the instructions below to begin traning our model.

```
python3 train.py --exp 'save/path/to/your/folder'
```

### 1.3 Testing

Follow the instructions below to begin testing our model.
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
