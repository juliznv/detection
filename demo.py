import os
import torch
import cv2
import numpy as np
import torchvision.transforms as tfs
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

from models.efficientnet import EfficientNet
from models.efficientdet import EfficientDet
from models.loss import FocalLoss

mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
#model = EfficientNet.from_pretrained('efficientnet-b0',False,True)
parameters = torch.load('weights/efficientnet-b0.pth')
model = EfficientNet.from_name('efficientnet-b0')
parameters = [v for _,v in parameters.items()]
model_state_dict = model.state_dict()
for i, (k,v) in enumerate(model_state_dict.items()):
    model_state_dict[k] = parameters[i]
torch.save(model_state_dict, 'weights/efficientnet-b0.pth')


'''
data_dir = 'data'
imgs_path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

imgs = [cv2.resize((cv2.imread(i)[...,::-1]/255 - mean)/std,(608,608)) for i in imgs_path]
imgs = torch.stack([torch.from_numpy(i.astype(np.float32)) for i in imgs], 0).permute(0, 3, 1, 2)

imgs = imgs[:4].cuda()
features = model.extract_features(imgs)
print(features.size())
'''