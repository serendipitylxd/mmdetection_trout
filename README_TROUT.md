<img src="docs/open_mmlab.png" align="right" width="30%">

# TROUT
`Trout` is a multi-modal data set of waterway traffic. This document provides how to use trout data set for 2d object detection training and evaluation. 

**Highlights**: 
* `TROUT` has been published, (Jan. 2025)
* The code already supports train, val, and test in `TROUT` dataset.


## Overview
- [Changelog](#changelog)
- [Model Zoo](#model-zoo)
- [Installation](#Installation)
- [Getting Started](#Getting-Started)
- [Citation](#citation)


## Changelog
[2025-01-19] `TROUT` v0.1.0 is released. 


## Introduction
The `TROUT` dataset contains the following specific attributes:
1) Data from the same location at different periods (including morning, noon, and afternoon) and different weather (including sunny, cloudy, and drizzly) scenarios;
2) 16000 frames of UAV image data, multi-LiDAR point cloud data, water depth data, etc. The real-time difference of the same frame data from different sensors is less than 0.1s;
3) 16000 frames of data of 85216 targets with detailed labeling information, including water level, 2D and 3D boxes of targets, and other labeling information.
The format of the dataset and annotation information is modeled after the Coco, Kitti, and PCDet_custom datasets, which is convenient for other researchers to get a better handle on the TROUT dataset and quickly conduct related research.
DATA at：[Coco_TROUT_data](https://drive.google.com/file/d/1mFQS-TTOR1sSjfPCuIeryK_LND2CLhk9/view?usp=sharing), [Kitti_TROUT_data](https://drive.google.com/file/d/1E5cQHYgv8s7pCfQyvnH2FvPhxDPe2lxb/view?usp=sharing), [PCDet_TROUT_data](https://drive.google.com/file/d/1JCClYd6egTm0AxXtW80L18AySgG45_og/view?usp=sharing).
<p align="center">
  <img src="docs/trout.png" width="95%">
</p>

* Trout dataset coordinate system
<p align="center">
  <img src="docs/trout_coordinates.png" width="95%">
</p>


## Model Zoo

### TROUT 2D Object Detection Baselines
Selected supported methods are shown in the below table. The results are the 2D detection performance on the `val`, `test` set of TROUT dataset.
* All models are trained with GTX 4080 GPU and are available for download. 
* The training time is measured with GTX 4080 GPU and PyTorch 2.5.1.
* Intersection over Union (IoU) from 0.5 to 0.95.

|                                             | training time | mAP(val) | mAP(test) | mAR(val) | mAR(test) | download | 
|---------------------------------------------|----------:|:-------:|:-------:|:-------:|:---------:|:---------:|
| [Atss](configs/atss/atss_r50_fpn_1x_coco.py) |~1.3 hours| 98.8 | 98.6 | 99.3 | 992 | [atss_trout_261M](https://drive.google.com/file/d/1EhawdW_1tQQAZ-QrreHErOwNxLKmphuP/view?usp=sharing) | 
| [Convnext](configs/convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py)       |  ~5.6 hours  | 98.9 | 98.7 | 99.4 | 99.1 | [convnext_trout_1125M](https://drive.google.com/file/d/1FyL_ZEuqRl9GxsvgX6Y6Guj0ezOuLvf6/view?usp=sharing) |
| [Dab_detr](configs/dab_detr/dab-detr_r50_8xb2-50e_coco.py) | ~4.5 hours | 92.9 | 93.0 | 95.6 | 95.5 | [dab_detr_trout_541M](https://drive.google.com/file/d/1k98gA-4NNs8dHuR6qhw5dDHsvJ855uf_/view?usp=sharing)| 
| [Ddq](configs/ddq/ddq-detr-4scale_r50_8xb2-12e_coco.py)    | ~3.9 hours| 98.9 | 98.8| 994 | 99.5 | [ddq_trout_621M](https://drive.google.com/file/d/1098VkIlZuPhmdg4TTM0XuN-t1I3Fb5tC/view?usp=sharing) |
| [Dino](tools/test.py configs/dino/dino-4scale_r50_8xb2-12e_coco.py) | ~2.1 hours| 98.8 | 98.8 | 99.4 | 99.4 | [dino_trout_600M](https://drive.google.com/file/d/19gbAcmZjLBu9h8PaDtQMqKo0zxPi_ksi/view?usp=sharing) |
| [yolox](configs/yolox/yolox_s_8xb8-300e_coco.py) | ~13.6 hours| 96.9 | 96.7 | 97.8 | 97.8 | [yolox_trout_142M](https://drive.google.com/file/d/1EN7vk39We_iZeQZ0HCwR2M3NkxgN_xgi/view?usp=sharing) |

## Installation
The installation process is the same as that for `MMDetection`.If you have problems with installation, you can refer to our conda environment. Our GPU is RTX4080 and the operating system is ubuntu 20.04.


Create and activate a conda environment.
```shell
conda create --name mmdet python=3.9 -y
conda activate mmdet
```

Install `PyTorch` based on the `PyTorch` official instructions.The cuda for our environment is 12.1, cuda for our environment is 12.1 and python is 3.9.
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

Install `MMEngine` and `MMCV` using `MIM`.
```shell
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
```

Install `MMdetection`.
```shell
git clone https://github.com/luxiaodong/trout.git
cd mmdetection_trout
pip install -v -e . 
```
Verify that the installation is successful. If it is successful, you can find `demo.jpg` in the vis folder.

```shell
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```

```
mmdetection_trout
├── outputs
│   ├── vis
│   │   │── demo.jpg
```

<p align="leftr">
  <img src="docs/demo.jpg" width="25%">
</p>


### conda list
```
# packages in environment at /home/luxiaodong/miniconda3/envs/mmdet:
#
# Name                    Version
_libgcc_mutex             0.1                 conda_forge    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
_openmp_mutex             4.5                       2_gnu    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
addict                    2.4.0                    pypi_0    pypi
aliyun-python-sdk-core    2.16.0                   pypi_0    pypi
aliyun-python-sdk-kms     2.16.5                   pypi_0    pypi
bzip2                     1.0.8                h4bc722e_7    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
ca-certificates           2024.12.14           hbcca054_0    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
certifi                   2024.12.14               pypi_0    pypi
cffi                      1.17.1                   pypi_0    pypi
charset-normalizer        3.4.1                    pypi_0    pypi
click                     8.1.8                    pypi_0    pypi
colorama                  0.4.6                    pypi_0    pypi
contourpy                 1.3.0                    pypi_0    pypi
crcmod                    1.7                      pypi_0    pypi
cryptography              44.0.0                   pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
einops                    0.8.0                    pypi_0    pypi
filelock                  3.14.0                   pypi_0    pypi
fonttools                 4.55.3                   pypi_0    pypi
fsspec                    2024.2.0                 pypi_0    pypi
idna                      3.10                     pypi_0    pypi
importlib-metadata        8.5.0                    pypi_0    pypi
importlib-resources       6.5.2                    pypi_0    pypi
jinja2                    3.1.3                    pypi_0    pypi
jmespath                  0.10.0                   pypi_0    pypi
kiwisolver                1.4.7                    pypi_0    pypi
ld_impl_linux-64          2.43                 h712a8e2_2    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
libffi                    3.4.2                h7f98852_5    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
libgcc                    14.2.0               h77fa898_1    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
libgcc-ng                 14.2.0               h69a702a_1    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
libgomp                   14.2.0               h77fa898_1    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
liblzma                   5.6.3                hb9d3cd8_1    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
libnsl                    2.0.1                hd590300_0    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
libsqlite                 3.47.2               hee588c1_0    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
libuuid                   2.38.1               h0b41bf4_0    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
libxcrypt                 4.4.36               hd590300_1    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
libzlib                   1.3.1                hb9d3cd8_2    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
markdown                  3.7                      pypi_0    pypi
markdown-it-py            3.0.0                    pypi_0    pypi
markupsafe                2.1.5                    pypi_0    pypi
mat4py                    0.6.0                    pypi_0    pypi
matplotlib                3.9.4                    pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
mmcv                      2.1.0                    pypi_0    pypi
mmdet                     3.3.0                     dev_0    <develop>
mmengine                  0.10.5                   pypi_0    pypi
mmpretrain                1.2.0                     dev_0    <develop>
model-index               0.1.11                   pypi_0    pypi
modelindex                0.0.2                    pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
ncurses                   6.5                  he02047a_1    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
networkx                  3.2.1                    pypi_0    pypi
numpy                     1.26.3                   pypi_0    pypi
nvidia-cublas-cu12        12.1.3.1                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-runtime-cu12  12.1.105                 pypi_0    pypi
nvidia-cudnn-cu12         9.1.0.70                 pypi_0    pypi
nvidia-cufft-cu12         11.0.2.54                pypi_0    pypi
nvidia-curand-cu12        10.3.2.106               pypi_0    pypi
nvidia-cusolver-cu12      11.4.5.107               pypi_0    pypi
nvidia-cusparse-cu12      12.1.0.106               pypi_0    pypi
nvidia-nccl-cu12          2.21.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.1.105                 pypi_0    pypi
nvidia-nvtx-cu12          12.1.105                 pypi_0    pypi
opencv-python             4.10.0.84                pypi_0    pypi
opendatalab               0.0.10                   pypi_0    pypi
openmim                   0.3.9                    pypi_0    pypi
openssl                   3.4.0                h7b32b05_1    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
openxlab                  0.1.2                    pypi_0    pypi
ordered-set               4.1.0                    pypi_0    pypi
oss2                      2.17.0                   pypi_0    pypi
packaging                 24.2                     pypi_0    pypi
pandas                    2.2.3                    pypi_0    pypi
pillow                    10.2.0                   pypi_0    pypi
pip                       24.3.1             pyh8b19718_2    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
platformdirs              4.3.6                    pypi_0    pypi
pycocotools               2.0.8                    pypi_0    pypi
pycparser                 2.22                     pypi_0    pypi
pycryptodome              3.21.0                   pypi_0    pypi
pygments                  2.19.1                   pypi_0    pypi
pyparsing                 3.2.1                    pypi_0    pypi
python                    3.9.21          h9c0c6dc_1_cpython    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
python-dateutil           2.9.0.post0              pypi_0    pypi
pytz                      2023.4                   pypi_0    pypi
pyyaml                    6.0.2                    pypi_0    pypi
readline                  8.2                  h8228510_1    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
requests                  2.28.2                   pypi_0    pypi
rich                      13.4.2                   pypi_0    pypi
scipy                     1.13.1                   pypi_0    pypi
setuptools                60.2.0                   pypi_0    pypi
shapely                   2.0.6                    pypi_0    pypi
six                       1.17.0                   pypi_0    pypi
sympy                     1.13.1                   pypi_0    pypi
tabulate                  0.9.0                    pypi_0    pypi
termcolor                 2.5.0                    pypi_0    pypi
terminaltables            3.1.10                   pypi_0    pypi
tk                        8.6.13          noxft_h4845f30_101    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
tomli                     2.2.1                    pypi_0    pypi
torch                     2.5.1+cu121              pypi_0    pypi
torchaudio                2.5.1+cu121              pypi_0    pypi
torchvision               0.20.1+cu121             pypi_0    pypi
tqdm                      4.65.2                   pypi_0    pypi
triton                    3.1.0                    pypi_0    pypi
typing-extensions         4.9.0                    pypi_0    pypi
tzdata                    2024.2                   pypi_0    pypi
urllib3                   1.26.20                  pypi_0    pypi
wheel                     0.45.1             pyhd8ed1ab_1    https://mirrors.ustc.edu.cn/anaconda/cloud/conda-forge
yapf                      0.43.0                   pypi_0    pypi
zipp                      3.21.0                   pypi_0    pypi
```

## Getting Started

### How to use TROUT data sets for training and evaluation
Train a model.

```shell
python tools/train.py <config_file> --work_dir <work_dir>
```
Give an example.

```shell
python tools/train.py configs/atss/atss_r50_fpn_1x_coco.py --work-dir atss_outputs  
```


Test and evaluate the pretrained modelsTest and evaluate the pretrained models.

```shell
python tools/test.py <config_file>  <pth_file>
```

Give an example.

```shell
python tools/test.py configs/atss/atss_r50_fpn_1x_coco.py atss_outputs/epoch_12.pth 
```

### What code we modified in the mmdetection environment
`mmdetection` is trained and evaluated with val data set instead of test data set. We modify the code `init.py`, `coco_detection.py` and `coco_instance.py` so that TROUT data set can be trained and evaluated under mmdetection environment, and the evaluation includes the test data of the dataset.


```
mmdetection_trout
├── mmdet
│   ├── datasets
│   │   │── coco.py
```

```python
    METAINFO = {
        'classes':
        ('Building', 'Fully_loaded_cargo_ship', 'Fully_loaded_container_ship', 'Lock_gate', 'Tree', 'Unladen_cargo_ship'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(255, 96, 55), (250, 50, 83), (255, 0, 124), (51, 221, 255), (221, 255, 51),
         (52, 209, 183)]
    }    
```

```
mmdetection_trout
├── cinfigs
│   ├── _base_
│   │   │── datasets
│   │   │   │── coco_detection.py
```

```

# dataset settings
dataset_type = 'CocoDataset'
#data_root = 'data/coco/'
data_root = 'data/coco_TROUT/'
```
```
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    #dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    #dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

```


```python
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
```

```python
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
```

```
mmdetection_trout
├── cinfigs
│   ├── _base_
│   │   │── datasets
│   │   │   │── coco_instance.py
```


```python
# dataset settings
dataset_type = 'CocoDataset'
#data_root = 'data/coco/'
data_root = 'data/coco_TROUT/'
```
```
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    #dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    #dict(type='Resize', scale=(1333, 800), keep_ratio=True),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
```


```python
test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))
```

```python
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test2017.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args)
```

#### atss
```
mmdetection_trout
├── cinfigs
│   ├── configs 
│   │   │── atss
│   │   │   │── atss_r50_fpn_1x_coco.py
```

```python
        #num_classes=80, 
        num_classes=6, ## Modify the network output dimension       
```

#### convnext 
```
mmdetection_trout
├── cinfigs
│   ├── configs 
│   │   │── convnext
│   │   │   │── cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py

```
```python
        #num_classes=80, 
        num_classes=6,         
```


```
        #num_classes=80, 
        num_classes=6,         
```


```
        #num_classes=80, 
        num_classes=6,        
```

```python
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(544, 1024), (592, 1024), (640, 1024), (688, 1024),
                        (736, 1024), (784, 1024), (832, 1024), (880, 1024),
                        (928, 1024), (976, 1024), (1024, 1024)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1024), (500, 1024), (600, 1024)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(544, 1024), (592, 1024), (640, 1024),
                                    (688, 1024), (736, 1024), (784, 1024),
                                    (832, 1024), (880, 1024), (928, 1024),
                                    (976, 1024), (1024, 1024)],
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]
```

#### dab_detr 
Download the dab-detr pre-training model.Put it in the dab_detr folder.

[dab-detr_r50_8xb2-50e_coco_20221122_120837-c1035c8c.pth](https://download.openmmlab.com/mmdetection/v3.0/dab_detr/dab-detr_r50_8xb2-50e_coco/dab-detr_r50_8xb2-50e_coco_20221122_120837-c1035c8c.pth) 


```
mmdetection_trout
├── cinfigs
│   ├── configs 
│   │   │── dab_detr
│   │   │   │── dab-detr_r50_8xb2-50e_coco_20221122_120837-c1035c8c.pth
```

```python
mmdetection_trout
├── cinfigs
│   ├── configs 
│   │   │── dab_detr
│   │   │   │── dab-detr_r50_8xb2-50e_coco.py
```

```python
#Load the pre-trained model   
load_from = './dab-detr_r50_8xb2-50e_coco_20221122_120837-c1035c8c.pth'

```

```python
        #num_classes=80, 
        num_classes=6,         
```


```python
        #num_classes=80, 
        num_classes=6,         
```


```python
        #num_classes=80, 
        num_classes=6,        
```

```python
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(544, 1024), (592, 1024), (640, 1024), (688, 1024),
                        (736, 1024), (784, 1024), (832, 1024), (880, 1024),
                        (928, 1024), (976, 1024), (1024, 1024)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1024), (500, 1024), (600, 1024)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(544, 1024), (592, 1024), (640, 1024),
                                    (688, 1024), (736, 1024), (784, 1024),
                                    (832, 1024), (880, 1024), (928, 1024),
                                    (976, 1024), (1024, 1024)],
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]
```
#### ddq
```
mmdetection_trout
├── cinfigs
│   ├── configs 
│   │   │── ddq
│   │   │   │── ddq-detr-4scale_r50_8xb2-12e_coco.py

```
```python
        #num_classes=80, 
        num_classes=6,      
```

```python
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                scales=[(544, 1024), (592, 1024), (640, 1024), (688, 1024),
                        (736, 1024), (784, 1024), (832, 1024), (880, 1024),
                        (928, 1024), (976, 1024), (1024, 1024)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                            scales=[(544, 1024), (592, 1024), (640, 1024),
                                    (688, 1024), (736, 1024), (784, 1024),
                                    (832, 1024), (880, 1024), (928, 1024),
                                    (976, 1024), (1024, 1024)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]
```
#### dino
```
mmdetection_trout
├── cinfigs
│   ├── configs 
│   │   │── dino
│   │   │   │── dino-4scale_r50_8xb2-12e_coco.py

```

```python
        #num_classes=80, 
        num_classes=6, 
        
```

```python
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                scales=[(544, 1024), (592, 1024), (640, 1024), (688, 1024),
                        (736, 1024), (784, 1024), (832, 1024), (880, 1024),
                        (928, 1024), (976, 1024), (1024, 1024)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    # The radio of all image in train dataset < 7
                    # follow the original implement
                    scales=[(400, 4200), (500, 4200), (600, 4200)],
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                            scales=[(544, 1024), (592, 1024), (640, 1024),
                                    (688, 1024), (736, 1024), (784, 1024),
                                    (832, 1024), (880, 1024), (928, 1024),
                                    (976, 1024), (1024, 1024)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]
```


#### yolox
```
mmdetection_trout
├── cinfigs
│   ├── configs 
│   │   │── yolox
│   │   │   │── yolox_s_8xb8-300e_coco.py
```

```python
        #num_classes=80, 
        num_classes=6, 
```

```python
# dataset settings
data_root = 'data/coco_TROUT/'
dataset_type = 'CocoDataset'
```

```python
test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/instances_test2017.json',
        data_prefix=dict(img='test2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_val2017.json',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/instances_test2017.json',
    metric='bbox',
    backend_args=backend_args)
```





## License
`TROUT` is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement
`TROUT` is an open source data set, and `TROUT` in the `MMdetection` environment is used for LiDAR based 2D scene perception, Multiple image-based detection models are supported. Some parts are learned from the official release code that supports the method above.
We would like to thank them for their proposed approach and formal implementation.

We hope that this repo will serve as a powerful and flexible code base that will benefit the research community by accelerating the process of re-implementing previous work and/or developing new methods.


## Citation 
If you find this project useful in your research, please consider cite:

```
@article{mmdetection,
  title   = {{MMDetection}: Open MMLab Detection Toolbox and Benchmark},
  author  = {Chen, Kai and Wang, Jiaqi and Pang, Jiangmiao and Cao, Yuhang and
             Xiong, Yu and Li, Xiaoxiao and Sun, Shuyang and Feng, Wansen and
             Liu, Ziwei and Xu, Jiarui and Zhang, Zheng and Cheng, Dazhi and
             Zhu, Chenchen and Cheng, Tianheng and Zhao, Qijie and Li, Buyu and
             Lu, Xin and Zhu, Rui and Wu, Yue and Dai, Jifeng and Wang, Jingdong
             and Shi, Jianping and Ouyang, Wanli and Loy, Chen Change and Lin, Dahua},
  journal= {arXiv preprint arXiv:1906.07155},
  year={2019}
}
```

```
@misc{trout2025,
    title={TROUT: Multi-Modal Dataset for Intelligent Waterway Traffic Monitoring Using UAV and LiDAR Integration},
    author={Xiaodong Lu, Weikai Tan,Kaofan Liu, Xinyue Luo,Sudong Xu},
    howpublished = {\url{https://github.com/serendipitylxd/mmdetection_trout}},
    year={2025}
}
```


## Contribution
Welcome to be a member of the `TROUT` development team by contributing to this repo, and feel free to contact us for any potential contributions. 


