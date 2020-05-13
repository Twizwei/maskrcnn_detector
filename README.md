# maskrcnn_detector
This repo is trying to implement the pipeline proposed by the paper [3D Bounding Box Estimation Using Deep Learning and Geometry](https://arxiv.org/pdf/1612.00496.pdf). Thanks to [Hongyuan](https://github.com/dashidhy) for implementing [deep 3D bbox](https://github.com/dashidhy/3D-BBox).

## Step 1: Download

Clone this repo.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
git clone https://github.com/Twizwei/maskrcnn_detector
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Step 2: Install maskrcnn-benchmark
Please follow the instructions [here](https://github.com/facebookresearch/maskrcnn-benchmark) to install the maskrcnn-benchmark for 2D object detection.

## Step 3: Install dependency for 3D-Bbox
```
pip install tensorboardx future
```
(update needed...)

## Step 3: Prepare dataset.
Currently we provide a data loader for [Kitti](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d).

Download the left color images, calibrations and labels.

The dataset should be organized as follows:
```
Kitti/
- training/
    - calib/
    - image_2/
    - label_2/
- testing/
    - calib/
    - image_2/
```

To run 2D object detector, you can create symbolic links under ```maskrcnn_detector/```:
```
mkdir datasets
cd datasets/
ln -s path-to-kitti-training ./
ln -s path-to-kitti-testing ./ 
```
Feel free to split the whole training set into training set and validation set.

To train MultiBins, you need to extract boxes from the labels.
```
cd Bbox_3d/datasets/kitti/
python make_kitti_box_set.py --root 'path-to-kitii'
```

## Step 4: Train 2D object detector and Deep 3D Bbox.
For maskrcnn, you can follow the [instructions](https://github.com/facebookresearch/maskrcnn-benchmark) to start training.

For Deep 3D Bbox, 
```
cd Bbox_3d/
python train.py --cfg_file 'path-to-config-file' --log_dir 'path-to-log-directory' --kitti_root 'path-to-kitii' --batch_size 4 --num_workers 4
```

## Step 5: Use the pipeline for inference
After training, you can use the pipeline for inference. For instance:
```
python tools/inference.py --mb_cfg_file ./Bbox_3d/configs/posenet_v0.py --weights path-to-detector-weights --config-file "./configs/e2e_faster_rcnn_R_50_C4_1x.yaml" DATASETS.TEST '("kitti_val",)' MODEL.ROI_BOX_HEAD.NUM_CLASSES 8
```
Feel free to modify the parameters.
