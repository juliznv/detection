"""
COCO-Style Evaluations
dataset structure
dataset_path/
    -project_name/
        -annotations/
            -instances_{set_name}.json
        -set_name/
            -*.jpg
"""

import json
import os

import argparse
import torch
import yaml
import torchvision.transforms as tfs
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader

from models import EfficientDet
from models.utils import BBoxTransform, ClipBoxes
from utils import preprocess, invert_affine, postprocess, boolean_string
from datasets import COCODetect

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--data-dir', type=str, help='path of datasets')
ap.add_argument('-p', '--project', type=str, default='coco',
                help = 'project file that contains datasets parameters (default:coco)')
ap.add_argument('-c', '--compound_coef', type=int, default=0, 
                help = 'coefficients of efficientdet compound (default: 0)')
ap.add_argument('-w', '--weights', type=str, default=None, 
                help='path of weights of model, like *.pth')
ap.add_argument('--nms_threshold', type=float, default=0.5, 
                help = 'nms threshold ( default: 0.5)')
ap.add_argument('--cuda', type = boolean_string, default = True, 
                help = 'use cuda or not (defualt: True)')
ap.add_argument('-bs', '--batch-size', type = int, default = 16,
                help = 'mini batch size (default: 16)')
ap.add_argument('--device', type = int, default = 0, 
                help = 'which device to use (default: 0)')
ap.add_argument('--float16', type = boolean_string, default = False, 
                help = 'use float16 or not (default: Flase)')
ap.add_argument('--override', type=boolean_string, default=True, 
                help='override previous results if exists (default: False)')
args = ap.parse_args()

data_dir=args.data_dir
compound_coef = args.compound_coef
nms_threshold = args.nms_threshold
use_cuda = args.cuda
gpu = args.device
use_float16 = args.float16
override_prev_results = args.override
project_name = args.project
weights_path = f'weights/efficientdet-d{compound_coef}.pth' if args.weights is None else args.weights
batch_size = args.batch_size
print(f'running coco-style evaluation on project {project_name}, weights {weights_path}...')

params = yaml.safe_load(open(f'configs/{project_name}.yml'))
obj_list = params['obj_list']

input_sizes = [512, 640, 768, 896, 1024, 1280, 1280, 1536, 1536]


def evaluate_coco(img_path, set_name, image_ids, coco, model, threshold=0.05):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for image_id in tqdm(image_ids):
        image_info = coco.loadImgs(image_id)[0]
        image_path = img_path + image_info['file_name']

        ori_imgs, framed_imgs, framed_metas = preprocess(image_path, max_size=input_sizes[compound_coef])
        x = torch.from_numpy(framed_imgs[0])

        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        x = x.unsqueeze(0).permute(0, 3, 1, 2)
        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        if not preds:
            continue

        preds = invert_affine(framed_metas, preds)[0]

        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']

        if rois.shape[0] > 0:
            # x1,y1,x2,y2 -> x1,y1,w,h
            rois[:, 2] -= rois[:, 0]
            rois[:, 3] -= rois[:, 1]

            bbox_score = scores

            for roi_id in range(rois.shape[0]):
                score = float(bbox_score[roi_id])
                label = int(class_ids[roi_id])
                box = rois[roi_id, :]

                image_result = {
                    'image_id': image_id,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': box.tolist(),
                }

                results.append(image_result)

    if not len(results):
        raise Exception('the model does not provide any valid output,\
            check model architecture and the data input')

    # write output
    filepath = f'efficientdet_d{compound_coef}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

def evaluate(dataloader, coco, model, threshold=0.5):
    results = []

    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()

    for i, data in tqdm(enumerate(dataloader)):
        x, img_ids, metas = data
        metas = metas.numpy().tolist()
        img_ids = img_ids.numpy().tolist()
        _,_,h,w = x.size()
        metas = [[w,h,ow,oh] for oh, ow in metas]
        if use_cuda:
            x = x.cuda(gpu)
            if use_float16:
                x = x.half()
            else:
                x = x.float()
        else:
            x = x.float()

        features, regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            regressBoxes, clipBoxes,
                            threshold, nms_threshold)
        
        if not preds:
            continue

        preds = invert_affine(metas, preds)
        for j, pred in enumerate(preds):
            scores = pred['scores']
            class_ids = pred['class_ids']
            rois = pred['rois']

            if rois.shape[0] > 0:
                # x1,y1,x2,y2 -> x1,y1,w,h
                rois[:, 2] -= rois[:, 0]
                rois[:, 3] -= rois[:, 1]

                bbox_score = scores

                for roi_id in range(rois.shape[0]):
                    score = float(bbox_score[roi_id])
                    label = int(class_ids[roi_id])
                    box = rois[roi_id, :]

                    image_result = {
                        'image_id': img_ids[j],
                        'category_id': label + 1,
                        'score': float(score),
                        'bbox': box.tolist(),
                    }

                    results.append(image_result)
                    
    if not len(results):
        raise Exception('the model does not provide any valid output,\
            check model architecture and the data input')

    # write output
    filepath = f'efficientdet_d{compound_coef}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(results, open(filepath, 'w'), indent=4)

def _eval(coco_gt, image_ids, pred_json_path):
    # load results in COCO evaluation tool
    coco_pred = coco_gt.loadRes(pred_json_path)

    # run COCO evaluation
    print('BBox')
    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == '__main__':
    SET_NAME = params['val_set']
    VAL_GT = f'{data_dir}/{params["project_name"]}/annotations/instances_{SET_NAME}.json'
    VAL_IMGS = f'{data_dir}/{params["project_name"]}/{SET_NAME}/'
    MAX_IMAGES = 10000
    img_tfs = tfs.Compose([
        tfs.Resize((input_sizes[compound_coef],input_sizes[compound_coef])),
        tfs.ToTensor(),
        tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    coco_gt = COCO(VAL_GT)
    image_ids = coco_gt.getImgIds()[:MAX_IMAGES]
    valset = COCODetect(f'{data_dir}/{params["project_name"]}',SET_NAME,img_tfs)
    valloader = DataLoader(valset, batch_size)
    
    
    if override_prev_results or not os.path.exists(f'{SET_NAME}_bbox_results.json'):
        model = EfficientDet(compound_coef=compound_coef, num_classes=len(obj_list),
                                     ratios=eval(params['anchors_ratios']), scales=eval(params['anchors_scales']))
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        model.requires_grad_(False)
        model.eval()

        if use_cuda:
            model.cuda(gpu)

            if use_float16:
                model.half()
        evaluate(valloader, coco_gt, model)
        #evaluate_coco(VAL_IMGS, SET_NAME, image_ids, coco_gt, model)

    _eval(coco_gt, image_ids, f'efficientdet_d{compound_coef}_bbox_results.json')
