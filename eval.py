import os
import argparse
import yaml
import json
import torch
import torchvision.transforms as tfs
import numpy as np

from PIL import Image
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models import EfficientDet

def arg_get():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type = str, help = 'path of data to eval.')
    parser.add_argument('-p', '--project', type = str, default = 'coco', 
                        help = 'project name (default: coco)')
    parser.add_argument('-c', '--compound', type = int, default = 0, 
                            help = 'compound coefficients of the EfficientDets.')
    parser.add_argument('-w', '--weights',type = str)
    args = parser.parse_args()
    return args

img_tfs = tfs.Compose([
    tfs.Resize((512,512)),
    tfs.ToTensor(),
    tfs.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


def post_process(imgs, anchors, regression, classfication,
                 regressBoxes, clipBoxes,
                 threshold, iou_threshold):
    transformed_anchors = regressBoxes(anchors, regression)
    transformed_anchors = clipBoxes(transformed_anchors, imgs)
    scores = torch.max(classfication,dim = 2, keepdim= True)[0]
    scores_over_threshold = (scores > threshold)[...,0]
    out = []
    for i in range(img.shape[0]):
        if scores_over_threshold[i].sum() == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
            continue
        c = classfication[i, scores_over_threshold[i,:],...].permute(1,0)
        ta = transformed_anchors[i, scores_over_threshold[i,:],...]
        s = scores[i,scores_over_threshold[i,:],...]
        s_, c_ = c.max(dim = 0)
        nms_idx = batch_nms(ta, s[:,0], c_, iou_threshold)
        if nms_idx.shape[0] == 0:
            out.append({
                'rois': np.array(()),
                'class_ids': np.array(()),
                'scores': np.array(()),
            })
        else:
            s_ = s_[nms_idx]
            c_ = c_[nms_idx]
            b_ = ta[nms_idx,:]

            
            out.append({
                'rois': b_.cpu().numpy(),
                'class_ids': c_.cpu().numpy(),
                'scores': s_.cpu().numpy(),
            })
        return out
        
def batch_nms(boxes, scores, idxs, iou_threshold):
    if boxes.numel() == 0:
        return torch.empty((0,),dtype= torch.int64,device= boxes.device)
    else:
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + torch.tensor(1).to(boxes))
        boxes_for_nms = boxes + offsets[:, None]
        keep = torch.ops.torchvision.nms(boxes_for_nms, scores, iou_threshold)
        return keep

def bbox_transform(anchors, regression):
    cy_a = (anchors[...,2] + anchors[...,0])/2
    cx_a = (anchors[...,3] + anchors[...,1])/2
    h_a = anchors[...,2] - anchors[...,0]
    w_a = anchors[...,3] - anchors[...,1]
    w = regression[...,3].exp() * w_a
    h = regression[...,2].exp() * h_a
    cy = regression[...,0] * w_a + cy_a
    cx = regression[...,1] * h_a + cx_a
    xmin = cx - w/2
    xmax = cx + w/2
    ymin = cy - h/2
    ymax = cy + h/2
    return torch.stack([xmin, ymin, xmax, ymax],dim = 2)

def clip_bbox(boxes, img):
    _, _, h, w = img.shape
    boxes[...,:2] = torch.clamp(boxes[...,:2], min = 0)
    boxes[...,2] = torch.clamp(boxes[...,2], max = w - 1)
    boxes[...,3] = torch.clamp(boxes[...,3], max = h - 1)

    return boxes

def invert_affine(metas,preds):
    for i in range(len(preds)):
        if len(preds[i]['rois']) != 0:
            if metas is float:
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / metas
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / metas
            else:
                o_h, o_w, n_h, n_w = metas[i]
                preds[i]['rois'][:, [0, 2]] = preds[i]['rois'][:, [0, 2]] / (n_w / o_w)
                preds[i]['rois'][:, [1, 3]] = preds[i]['rois'][:, [1, 3]] / (n_h / o_h)
    return preds



if __name__ == "__main__":
    args = arg_get()
    params = yaml.safe_load(open(f'configs/{args.project}.yml'))
    data_dir = os.path.join(args.data, args.project)
    set_name = params['val_set']
    ann_file = os.path.join(data_dir, 'annotations', f'instances_{set_name}.json')
    img_dir = os.path.join(args.data, args.project, set_name)
    coco = COCO(ann_file)
    img_ids = coco.getImgIds()
    model = EfficientDet(num_classes = len(params['obj_list']), compound_coef = args.compound)
    model.load_state_dict(torch.load(args.weights))
    model.requires_grad_(False)
    model.eval()
    model.cuda()

    result = []
    for i in tqdm(img_ids):
        h, w, img_path = [coco.loadImgs(i)[0][j] for j in ['height','width','file_name']]
        img = Image.open(os.path.join(img_dir, img_path)).convert("RGB")
        img = img_tfs(img)
        img = img.unsqueeze(0).cuda()
        features, regression, classfication, anchors = model(img)
        preds = post_process(img,anchors,regression,classfication,bbox_transform,clip_bbox,0.2,0.8)
        if not preds:
            continue
        preds = invert_affine([[h,w,*img.shape[2:]]],preds)[0]
        scores = preds['scores']
        class_ids = preds['class_ids']
        rois = preds['rois']
        if rois.shape[0] > 0:
            rois[:,2] -= rois[:,0]
            rois[:,3] -= rois[:,1]

            for roi_id in range(rois.shape[0]):
                score = float(scores[roi_id])
                label = int(class_ids[roi_id])
                bbox = rois[roi_id,:]
                img_result = {
                    'image_id': i,
                    'category_id': label + 1,
                    'score': float(score),
                    'bbox': bbox.tolist(),
                }
                result.append(img_result)
        
    if not len(result):
        raise Exception('the model does not provide any valid output, check model architecture and the data input')

    
    filepath = f'efficientdet_d{args.compound}_{set_name}_bbox_results.json'
    if os.path.exists(filepath):
        os.remove(filepath)
    json.dump(result, open(filepath, 'w'), indent=4)

    coco_pred = coco.loadRes(filepath)

    coco_eval = COCOeval(coco, coco_pred, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
