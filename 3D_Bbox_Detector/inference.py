# To infer 2d object bounding boxes.
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch

import Bbox_3d.models 
from Bbox_3d.miscs import config_utils as cd, eval_utils as eu, X_Logger

from maskrcnn_benchmark.config import config
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference # this could be modified
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir


def build_3d_net(args):
    # build MultiBins network
    pass

def prediction(args):
    """
    predict with 3D Bbox pipeline:
    2D object detection -> MultiBin -> Physical Constraints.
    """
    # Prepare args for MultiBins
    mb_cfg_dict = cu.file2dict(args.mb_cfg_file)
    mb_model_cfg = cfg_dict['model_cfg'].copy()

    # merge to MaskRCNN benchmark argparser.
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # Build data loader and load weights.
    faster_rcnn = build_detection_model(cfg)
    faster_rcnn.eval()
    model.to(cfg.MODEL.DEVICE)

    # Build MultiBins.
    multibins = build_3d_net(args)

    # Run Faster-Rcnn.

    # Run MultiBins

    pass

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
    parser.add_argument("--weights", type=str, nargs='+')
    
    # For MultiBins
    parser.add_argument('--mb_cfg_file', 
                        type=str, 
                        required=True, 
                        help='MultiBins Config file path, required flag.')
    parser.add_argument('--mb_log_dir', type=str, help='Folder to save experiment records.')
    parser.add_argument('--mb_batch_size', type=int, help='Mini-batch size.')
    parser.add_argument('--mb_num_workers', type=int, help='Number of workers.')
    
    args = parser.parse_args()
    prediction(args)