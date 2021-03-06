# To infer 2d object bounding boxes.
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
from tqdm import tqdm

import torch
from torchvision import transforms as T
import numpy as np
from PIL import Image 

import Bbox_3d.models as models
from Bbox_3d.miscs import config_utils as cu, eval_utils as eu, X_Logger
from Bbox_3d.estimate_3d_bbox_v2 import dimensions_to_corners, solve_3d_bbox_single

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference # this could be modified
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir

def box2patch(image_dir, detection_result):
    """
    Clip and get bbox patch, also normalize
    """
    img = Image.open(os.path.join(image_dir, detection_result['name']))
    # for normalization
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])
    box_patches = []
    for label in detection_result['labels']:
        score = label['score']
        if score < 0.2:
            continue
        box = label['box2d']
        box_patch = img.resize((224, 224), box=box)
        box_patch = transform(box_patch)
        box_patches.append(box_patch)
    
    box_patches = torch.stack(box_patches)
    return box_patches
    

def build_3d_net(model, config):
    # build posenet network
    posenet = models.builder.build_from(model, config)

    return posenet

def prediction(args):
    """
    predict with 3D Bbox pipeline:
    2D object detection -> PoseNet -> Physical Constraints.
    """
    # Prepare args for Posenet
    mb_cfg_dict = cu.file2dict(args.mb_cfg_file)
    mb_model_cfg = mb_cfg_dict['model_cfg'].copy()
    mb_loss_cfg = mb_cfg_dict['loss_cfg'].copy()

    # merge to MaskRCNN benchmark argparser.
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Build detector and load weights.
    detector = build_detection_model(cfg)
    detector.eval()
    detector.to(cfg.MODEL.DEVICE)
    checkpointer_detector = DetectronCheckpointer(cfg, detector)
    print(args.weights)
    _ = checkpointer_detector.load(args.weights)

    # Build Posenet.
    posenet = build_3d_net(models, mb_model_cfg).to(cfg.MODEL.DEVICE) # TODO: build and load weights
    posenet.eval()
    
    bin_centers = torch.arange(mb_loss_cfg['pose_loss_cfg']['num_bins']).float() * 2 * np.pi / mb_loss_cfg['pose_loss_cfg']['num_bins']
    bin_centers[bin_centers > np.pi] -= 2 * np.pi # to [-pi, pi]
    bin_centers = bin_centers.to(cfg.MODEL.DEVICE)

    # build data loader.
    output_dirs = [None] * len(cfg.DATASETS.TEST)
    if cfg.OUTPUT_DIR:
        dataset_names = cfg.DATASETS.TEST
        output_dir = cfg.OUTPUT_DIR
        mkdir(output_dir)
    data_loader = make_data_loader(cfg, is_train=False, is_distributed=False)  
    data_loader = data_loader[0].dataset
    data_loader.return_names = True  # TODO: make this look good
    image_dir = data_loader.image_dir

    cpu_device = torch.device("cpu")
    detection_results = []
    for i, batch in tqdm(enumerate(data_loader)):
        images, _, image_names, calib = batch  # Note: images are normalized
        
        images = images.unsqueeze(0)  # TODO
        
        images = images.to(cfg.MODEL.DEVICE)
        print('input image shape:', images.shape)
        # Run detector.
        with torch.no_grad():
            outputs = detector(images)  # TODO: list[BoxList], re-design output contents.
            outputs = [o.to(cpu_device) for o in outputs]
            # detection_results.update({img_id: result for img_id, result in zip(image_ids, output)})
        patches = []
        for output in outputs:  # for each boxlist (each single image)
            detection_result = {
                'name': image_names,
                'labels': [],
            }
            boxes = output.bbox.numpy().tolist()
            labels = output.get_field('labels').numpy().tolist()
            scores = output.get_field('scores').numpy().tolist()
            for k in range(len(boxes)):
                detection_result['labels'] += [{
                    'category': labels[k],
                    'box2d': (
                        boxes[k][0],
                        boxes[k][1],
                        boxes[k][2],
                        boxes[k][3]
                    ),
                    'score': scores[k]
                }]  # detection_result stores all boxes in a single image
            
            # Process 2D Bbox, generate patches. Currently single image per time.
            patches_single_img = box2patch(image_dir, detection_result)  # shape [num_boxes, 3, 224, 224]
            patches.append(patches_single_img)
            print('patch 0 shape:', (patches[0]).shape)

            # save 2d detection results, maybe useless
            detection_results += [detection_result]
            
        # patches to inputs, Run Posenet
        dim_preds = []
        orient_preds = []
        trans_preds = []
        for k in range(patches[0].shape[0]):  # TODO: currently one image per time
            input_patch_batch = patches[0][k, :].unsqueeze(0).to(cfg.MODEL.DEVICE)
            print('input patch shape:', input_patch_batch.shape)
            with torch.no_grad():
                dim_pred, bin_score, bin_pred = posenet(input_patch_batch)
                print('posenet output shape:', dim_pred.shape, bin_pred.shape, bin_score.shape)
            box2d = torch.tensor(boxes[k]).squeeze().to(cpu_device)
            corners = dimensions_to_corners(dim_pred).squeeze().to(cpu_device)
            bin_value = bin_pred + bin_centers
            bin_value[bin_value > np.pi] -= 2.0 * np.pi
            bin_value[bin_value < -np.pi] += 2.0 * np.pi
            print('bin value shape:', bin_value.shape)
            theta_l = torch.gather(bin_value, 1, torch.argmax(bin_score, dim=1, keepdim=True)).to(cpu_device)
            orient_preds.append(theta_l)
            print('theta_l:', theta_l)

            # Physical constraints, one object by one, TODO: make it parallel?
            for object_idx in range(bin_value.shape[0]):
                translation = solve_3d_bbox_single(box2d, corners, theta_l[object_idx], calib)
                trans_preds.append(translation)
                print('translation:', translation)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MaskRCNN Benchmark Inference")
    # For detector
    parser.add_argument(
        "--config-file",
        default="/home/selfdriving/maskrcnn_detector/configs/e2e_faster_rcnn_R_50_C4_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument("--weights", type=str)
    
    # For PoseNet
    parser.add_argument('--mb_cfg_file', 
                        type=str, 
                        required=True, 
                        help='Posenet Config file path, required flag.')
    parser.add_argument('--mb_log_dir', type=str, help='Folder to save experiment records.')
    # parser.add_argument('--mb_batch_size', type=int, help='Mini-batch size.')
    parser.add_argument('--mb_num_workers', type=int, help='Number of workers.')
    
    args = parser.parse_args()
    prediction(args)