import torch
import cv2
import yaml
import numpy as np

from models import EfficientDet, BBoxTransform, ClipBoxes
from utils import aspectaware_resize_padding,postprocess,invert_affine

def detector(img, model, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), cuda = True):
    with torch.no_grad():
        img_norm = (img / 255 - mean) / std
        nimg, nw, nh, ow, oh, _, _ = aspectaware_resize_padding(img, 512, 512)
        x = torch.from_numpy(nimg).permute(2,0,1).unsqueeze(0)
        if cuda:
            model = model.cuda()
            x = x.cuda()
        regression, classification, anchors = model(x)

        preds = postprocess(x,
                            anchors, regression, classification,
                            BBoxTransform(), ClipBoxes(),
                            0.5, 0.7)
        
        if not preds:
            return None
        preds = invert_affine([[nw, nh, ow, oh]], preds)
        print(len(preds[0]['rois']))
        return preds
        


params = yaml.safe_load(open('configs/coco.yml'))

if __name__ == '__main__':
    model = EfficientDet(compound_coef = 0, num_classes = len(params['obj_list']),
                         ratios = eval(params['anchors_ratios']),
                         scales = eval(params['anchors_scales']))
    model.load_state_dict(torch.load('./weights/efficientdet-d0.pth'))
    model.eval()
    img = cv2.imread('demo.jpg')
    out = detector(img, model)


    

