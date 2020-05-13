# maskrcnn_detector
For 3D BBox inputs. Thanks to [Hongyuan](https://github.com/dashidhy) for implementing [deep 3D bbox](https://github.com/dashidhy/3D-BBox).

This repo is trying to implement the pipeline proposed by the paper [3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/pdf/1612.00496.pdf).

## Step 1: Download

Clone this repo.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
git clone https://github.com/Twizwei/maskrcnn_detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Step 2: Install maskrcnn-benchmark
Please follow the instructions [here](https://github.com/facebookresearch/maskrcnn-benchmark) to install the maskrcnn-benchmark for 2D object detection.

## Step 3: Prepare dataset.
Currently we provide a data loader for [Kitti](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

Download the left color images, calibrations and labels.

The dataset should be organized as follows:
```
~/Kitti:
- training
    - calib
    - image_2
    - label_2
- testing
    - calib
    - image_2

```

