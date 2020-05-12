import argparse
import copy
import json
import os
from collections import defaultdict

import os.path as osp

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib import patches

from maskrcnn_benchmark.config import cfg

def parse_args():
    """Use argparse to get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--result', type=str, help='path to results to be evaluated')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    return args

def group_by_key(detections, key):
    groups = defaultdict(list)
    
    for d in detections:
        groups[d[key]].append(d)
    return groups

def main():
    args = parse_args()
    
    cfg.merge_from_list(args.opts)

    image_dir = '/home/selfdriving/faster-rcnn-KITTI-BDD100K/datasets/kitti/test/images/'
    for r in sorted(os.listdir(args.result)):
        if not r.endswith('.json'):
            continue
        print('evaluating {}...'.format(r))
        with open(os.path.join(args.result, r)) as f:
            result = json.load(f)
        
        fig_dir = os.path.join(args.result, 'figures' + r[:-5])

        class_id_dict = {
                        'pedestrian': 1,
                        'cyclist': 2,
                        'person_sitting': 3,
                        'car': 4,
                        'van': 5,
                        'truck': 6,
                        'tram': 7
                        }
                        
        cat_list = [class_id_dict[k] for k in class_id_dict]
        print(cat_list)
        for idx in range(len(result)):
            cat_pred = group_by_key(result[idx]['labels'], 'category')
            for i, cat in enumerate(cat_list):
                if cat in cat_pred:
                    if cat == 0:
                        print('cat: 0!')
                    if not os.path.exists(fig_dir):
                        os.mkdir(fig_dir)
                    print('save to..', fig_dir)
                    if len(fig_dir) > 0 and idx % 20 == 0:
                        fig, ax = plt.figure(), plt.gca()
                        # image = Image.open(os.path.join(image_dir, gt[idx]['name']))
                        image = Image.open(os.path.join(image_dir, result[idx]['name']))
                        ax.imshow(image)
                        # print(cat_pred)
                        flag = False
                        for l in cat_pred[cat]:
                            if l['score'] > 0.5:
                                x1, y1, x2, y2 = l['box2d']['x1'], l['box2d']['y1'], l['box2d']['x2'], l['box2d']['y2']
                                ax.add_patch(
                                    patches.Rectangle(
                                        (x1, y1),
                                        x2 - x1,
                                        y2 - y1,
                                        edgecolor='r',
                                        linewidth=1,
                                        fill=False
                                    )
                                )
                                flag = True
                        if flag:
                            plt.axis('scaled')
                            plt.tight_layout()
                            fn = '{}_{}_{}.jpg'.format(cat, idx, r[:-5])
                            plt.savefig(os.path.join(fig_dir, fn))
                            plt.close(fig)

if __name__ == '__main__':
    main()
