<div id="top" align="center">

# 3D occupancy Challenge

<img src="./figs/occupancy.gif" width="600px">

</div>

<div align="left">

## Table of Contents

- [Introduction](#introduction)
- [Model Zoo](#model-zoo)
- [Task Definition](#task-definition)
- [Evaluation Metrics](#evaluation-metrics)
- [OpenOcc Dataset](#openocc-dataset)
- [References](#references)

## Introduction

Understanding the 3D surroundings including the background stuffs and foreground objects is important for autonomous driving. In the traditional 3D object detection task, a foreground object is represented by the 3D bounding box. However, the geometrical shape of the object is complex, which can not be represented by a simple 3D box, and the perception of the background stuffs is absent. The goal of this task is to predict the 3D occupancy of the scene. In this task, we provide a large-scale occupancy benchmark based on the nuScenes dataset. The benchmark is a voxelized representation of the 3D space, and the occupancy state and semantics of the voxel in 3D space are jointly estimated in this task. The complexity of this task lies in the dense prediction of 3D space given the surround-view images.

# Model Zoo

The following table summarizes the performance metrics of different models tested on the validation dataset of Nuscenes dataset for 3d occupancy prediction.

<div align="center">

| Method                                     | Backbone | Config | RayIOU       | Weights       | Memory | FPS on A100 |
|--------------------------------------------|----------|--------|--------------|---------------|--------|-------|
| BEVFormer                                  |  ResNet50  | [config](https://github.com/saiprakash-c/3d-occupancy-prediction/blob/challenge/projects/configs/bevformer/bevformer_base_occ_pretrained.py)       | 0.285        |               |  15.8GB | ~3    |
| SparseBEV                                  | ResNet50   | [config](https://github.com/saiprakash-c/SparseOcc/blob/main/configs/r50_nuimg_704x256_8f_openocc.py) | 0.3312       | [weights](https://github.com/saiprakash-c/SparseOcc/blob/60d2c5ee930eeb1c3eb0f778b9fdd951727a88f4/configs/r50_nuimg_704x256_8f_openocc.py) | 20.7GB | ~17 |
| SparseBEV | FlashInternImage-T |  In progress |   |   |        | |

<div align="left">

## Testing

The installation instructions in the original repositories have dependency issues. Instead, pull the the docker `saiprakashc/mmdet3d-image:v0.17.0-2` for testing BEVFormer and the docker `saiprakashc/mmdet3d-image:v1.0.0rc6` for testing SparseOcc. The docker images exist in docker hub and they work right out of the box. 

For setting up the data, see [data](#openocc-dataset) section.

## Task Definition

Given images from multiple cameras, the goal is to predict the semantics and flow of each voxel grid in the scene.

## Evaluation Metrics

The implementation is here: [projects/mmdet3d_plugin/datasets/ray_metrics.py](https://github.com/OpenDriveLab/OccNet/blob/challenge/projects/mmdet3d_plugin/datasets/ray_metrics.py)

### Ray-based mIoU

We use the well-known mean intersection-over-union (mIoU) metric. However, the elements of the set are now query rays, not voxels.

Specifically, we emulate LiDAR by projecting query rays into the predicted 3D occupancy volume. For each query ray, we compute the distance it travels before it intersects any surface. We then retrieve the corresponding class label and flow prediction.

We apply the same procedure to the ground-truth occupancy to obtain the groud-truth depth, class label and flow.

A query ray is classified as a **true positive** (TP) if the class labels coincide **and** the L1 error between the ground-truth depth and the predicted depth is less than either a certain threshold (e.g. 2m).

Let $C$ be he number of classes. 

$$
mIoU=\frac{1}{C}\displaystyle \sum_{c=1}^{C}\frac{TP_c}{TP_c+FP_c+FN_c},
$$

where $TP_c$ , $FP_c$ , and $FN_c$ correspond to the number of true positive, false positive, and false negative predictions for class $c_i$.

We finally average over distance thresholds of {1, 2, 4} meters and compute the mean across classes.

For more details about this metric, please refer to the [technical report](https://arxiv.org/abs/2312.17118).

### AVE for Occupancy Flow

Here we measure velocity errors for a set of true positives (TP). We use a threshold of 2m distance.

The absolute velocity error (AVE) is defined for 8 classes ('car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'pedestrian') in m/s. 

### Occupancy Score

The final occupancy score is defined to be a weighted sum of mIoU and mAVE. Note that the velocity errors are converted to velocity scores as `max(1 - mAVE, 0.0)`. That is,

```
OccScore = mIoU * 0.9 + max(1 - mAVE, 0.0) * 0.1
```

<p align="right">(<a href="#top">back to top</a>)</p>

## OpenOcc Dataset

### Basic Information

- The **nuScenes OpenOcc** dataset contains 17 classes. Voxel semantics for each sample frame is given as `[semantics]` in the labels.npz. Occupancy flow is given as `[flow]`  in the labels.npz.

<div align="center">

|  Type  |  Info  |
| :----: | :----: |
| train       | 28,130 |
| val         | 6,019 |
| test        | 6,008 |
| cameras     | 6 |
| voxel size  | 0.4m |
| range       | [-40m, -40m, -1m, 40m, 40m, 5.4m] |
| volume size | [200, 200, 16] |
| #classes    | 0 - 16 |

</div>

### Download

1. Download the nuScenes dataset and put in into `data/nuscenes`

2. Download our `openocc_v2.1.zip` and `infos.zip` from [OpenDataLab](https://opendatalab.com/OpenDriveLab/CVPR24-Occ-Flow-Challenge/tree/main) or [Google Drive](https://drive.google.com/drive/folders/1lpqjXZRKEvNHFhsxTf0MOE13AZ3q4bTq)

3. Unzip them in `data/nuscenes`

### Hierarchy

The hierarchy of folder `data/nuscenes` is described below:

```
nuscenes
├── maps
├── nuscenes_infos_train_occ.pkl
├── nuscenes_infos_val_occ.pkl
├── nuscenes_infos_test_occ.pkl
├── openocc_v2
├── samples
├── v1.0-test
└── v1.0-trainval
```

- `openocc_v2` is the occuapncy GT.
- `nuscenes_infos_{train/val/test}_occ.pkl` contains meta infos of the dataset.
- Other folders are borrowed from the official nuScenes dataset.

### Known Issues

- nuScenes ([issue #721](https://github.com/nutonomy/nuscenes-devkit/issues/721)) lacks translation in the z-axis, which makes it hard to recover accurate 6d localization and would lead to the misalignment of point clouds while accumulating them over whole scenes. Ground stratification occurs in several data.

<p align="right">(<a href="#top">back to top</a>)</p>

## ToDo

- [ ] Train SparseBEV with FlashInternImage-T
- [ ] Evaluate FB-Occ
- [ ] Implement flow predicter for the best model

## References

```bibtex
@article{sima2023_occnet,
    title={Scene as Occupancy},
    author={Chonghao Sima and Wenwen Tong and Tai Wang and Li Chen and Silei Wu and Hanming Deng and Yi Gu and Lewei Lu and Ping Luo and Dahua Lin and Hongyang Li},
    year={2023},
    eprint={2306.02851},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

```bibtex
@misc{liu2024fully,
    title={Fully Sparse 3D Occupancy Prediction}, 
    author={Haisong Liu and Yang Chen and Haiguang Wang and Zetong Yang and Tianyu Li and Jia Zeng and Li Chen and Hongyang Li and Limin Wang},
    year={2024},
    eprint={2312.17118},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```

This dataset is under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license. Before using the dataset, you should register on the website and agree to the terms of use of the [nuScenes](https://www.nuscenes.org/nuscenes). 

<p align="right">(<a href="#top">back to top</a>)</p>
