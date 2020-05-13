# Adapted from https://github.com/ucbdrive/bdd-data/blob/master/bdd_data/evaluate.py

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
from maskrcnn_benchmark.data import make_data_loader

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


def fast_hist(gt, prediction, n):
    k = (gt >= 0) & (gt < n)
    return np.bincount(
        n * gt[k].astype(int) + prediction[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    ious = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    ious[np.isnan(ious)] = 0
    return ious


def find_all_png(folder):
    paths = []
    for root, dirs, files in os.walk(folder, topdown=True):
        paths.extend([osp.join(root, f)
                      for f in files if osp.splitext(f)[1] == '.png'])
    return paths


def get_ap(recalls, precisions):
    # correct AP calculation
    # first append sentinel values at the end
    recalls = np.concatenate(([0.], recalls, [1.]))
    precisions = np.concatenate(([0.], precisions, [0.]))

    # compute the precision envelope
    for i in range(precisions.size - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(recalls[1:] != recalls[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((recalls[i + 1] - recalls[i]) * precisions[i + 1])
    return ap


def group_by_key(detections, key):
    groups = defaultdict(list)
    
    for d in detections:
        groups[d[key]].append(d)
    return groups


def cat_pc(gt, predictions, thresholds):
    """
    Implementation refers to https://github.com/rbgirshick/py-faster-rcnn
    """
    predictions = sorted(predictions, key=lambda x: x['score'], reverse=True)
    gt_boxes = [[g['box2d']['x1'], g['box2d']['y1'], g['box2d']['x2'], g['box2d']['y2']] for g in gt]
    gt_boxes = np.array(gt_boxes)
    gt_checked = np.zeros((len(gt_boxes), len(thresholds)))

    # go down dets and mark TPs and FPs
    nd = len(predictions)
    tp = np.zeros((nd, len(thresholds)))
    fp = np.zeros((nd, len(thresholds)))
    for i, p in enumerate(predictions):
        box = p['box2d']
        x1, x2, y1, y2 = box['x1'], box['x2'], box['y1'], box['y2']
        ovmax = -np.inf
        jmax = -1
        
        if len(gt_boxes) > 0:
            # compute overlaps
            # intersection
            ixmin = np.maximum(gt_boxes[:, 0], x1)
            iymin = np.maximum(gt_boxes[:, 1], y1)
            ixmax = np.minimum(gt_boxes[:, 2], x2)
            iymax = np.minimum(gt_boxes[:, 3], y2)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((x2 - x1 + 1.) * (y2 - y1 + 1.) +
                   (gt_boxes[:, 2] - gt_boxes[:, 0] + 1.) *
                   (gt_boxes[:, 3] - gt_boxes[:, 1] + 1.) - inters)

            overlaps = inters / uni
            ovmax = np.max(overlaps)
            jmax = np.argmax(overlaps)

        for t, threshold in enumerate(thresholds):
            if ovmax > threshold:
                if gt_checked[jmax, t] == 0:
                    tp[i, t] = 1.
                    gt_checked[jmax, t] = 1
                else:
                    fp[i, t] = 1.
            else:
                fp[i, t] = 1.

    # compute precision recall
    fp = np.cumsum(fp, axis=0)
    tp = np.cumsum(tp, axis=0)
    recalls = tp / float(len(gt))
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = np.zeros(len(thresholds))
    for t in range(len(thresholds)):
        ap[t] = get_ap(recalls[:, t], precisions[:, t])
    
    if (np.max(recalls) > 1 or np.max(precisions) > 1 or np.max(ap) > 1):
        print(np.max(recalls), np.max(precisions), np.max(ap))
    
    return recalls, precisions, ap


def evaluate_detection(gt, pred, class_id_dict, fig_dir, model_name, image_dir, output_images=False):

    thresholds = [0.5, 0.75]
    aps = np.zeros((len(thresholds), len(class_id_dict.keys())))
    cat_list = [class_id_dict[k] for k in class_id_dict]
    print(cat_list)
    counters = np.zeros(len(cat_list))
    for idx in range(len(gt)):
        print('{}/{}'.format(idx, len(gt)))
        cat_gt = group_by_key(gt[idx]['labels'], 'category')
        cat_pred = group_by_key(pred[idx]['labels'], 'category')
        
        for i, cat in enumerate(cat_list):
            
            if cat in cat_pred and cat in cat_gt:
                if cat == 0:
                    print('cat 0!')
                r, p, ap = cat_pc(cat_gt[cat], cat_pred[cat], thresholds)
                
                aps[:, i] += ap
                counters[i] += 1
                
                
                if not os.path.exists(fig_dir):
                    os.mkdir(fig_dir)
                print('save to', fig_dir)
                if len(fig_dir) > 0 and idx % 200 == 0 and output_images:
                    fig, ax = plt.figure(), plt.gca()
                    # image = Image.open(os.path.join(image_dir, gt[idx]['name']))
                    image = Image.open(gt[idx]['name'])
                    ax.imshow(image)

                    for l in cat_pred[cat]:
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

                    for l in cat_gt[cat]:
                        x1, y1, x2, y2 = l['box2d']['x1'], l['box2d']['y1'], l['box2d']['x2'], l['box2d']['y2']
                        ax.add_patch(
                            patches.Rectangle(
                                (x1, y1),
                                x2 - x1,
                                y2 - y1,
                                edgecolor='g',
                                linewidth=1,
                                fill=False
                            )
                        )        

                    plt.axis('scaled')
                    plt.tight_layout()
                    fn = '{}_{}_{}.jpg'.format(cat, idx, model_name)
                    plt.savefig(os.path.join(fig_dir, fn))
                    plt.close(fig)
        

    for i in range(len(counters)):
        if counters[i] > 0:
            aps[:, i] /= counters[i]
    print(aps)
    mAP = np.mean(aps)
    return mAP, aps.flatten().tolist()


def main():
    args = parse_args()
    
    cfg.merge_from_list(args.opts)

    print('loading data...')
    data_loader = make_data_loader(cfg, is_train=False, is_distributed=False)[0]
    gt = data_loader.dataset.get_gt_labels()
    class_id_dict = data_loader.dataset.get_classes_ids()
    print('loaded!')
    best = 0
    best_json = ''
    for r in sorted(os.listdir(args.result)):
        if not r.endswith('.json'):
            continue
        print('evaluating {}...'.format(r))
        with open(os.path.join(args.result, r)) as f:
            result = json.load(f)
        
        fig_dir = os.path.join(args.result, 'figures')
        print(data_loader.dataset.image_dir)
        mean, breakdown = evaluate_detection(gt, result, class_id_dict, fig_dir, r[:-5], data_loader.dataset.image_dir, output_images=True)

        print('{:.2f}'.format(mean),
              ', '.join(['{:.2f}'.format(n) for n in breakdown]))
        
        if mean > best:
            best = mean
            best_json = r
        
        f = open(os.path.join(args.result, './eval' + str(r[:-5]) + '.txt'), 'w')
        f.write('ap50: ' + str(breakdown[1:8]) + ' mean: ' + str(np.mean(breakdown[1:8])) + '\n')
        f.write('ap75: ' + str(breakdown[8:]) + ' mean: ' + str(np.mean(breakdown[8:])) +'\n')
        #f.write('AP50: ' + str(mean) + '\n')
        f.write(str(best_json))
        f.close()
            
    print(best, best_json)
    

if __name__ == '__main__':
    main()
