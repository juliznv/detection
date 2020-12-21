import os
import argparse
import torch
import torch.nn as nn
import torchvision.transforms as tfs

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable
from PIL import Image
from tqdm import tqdm

from models import EfficientNet, EfficientDet

class Images(Dataset):
    '''
    An only image loader.
        root/*.png
    
    Args:
        root (str): Root directory path.
        transform (callable, optional): A function/transform that takes in an PIL image  
        and returns a transformed version.
    Attributes: 
        img_files (list[str]): List of image path.

    '''
    def __init__(self, root: str,
                 transform : Optional[Callable] = None,
                 ):
        super(Images, self).__init__()

        self.root = root
        self.transform = transform
        self.img_files = [f for f in os.listdir(root) if f.endswith('.JPEG')]

    def __getitem__(self, index):
        img_path = os.path.join(self.root, self.img_files[index])
        img = Image.open(open(img_path, 'rb')).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_files)

class EfficientNetFeatures(nn.Module):
    '''
    Get the feature maps of image by EfficientNet.

    Args:
        compound_coef (int): Compound coefficient of EfficientNet.
    '''
    def __init__(self, compound_coef):
        super(EfficientNetFeatures, self).__init__()
        
        model = EfficientNet.from_pretrained(f'efficientnet-b{compound_coef}', False)
        del model._conv_head
        del model._bn1
        del model._avg_pooling
        del model._dropout
        del model._fc
        self.model = model
    
    def forward(self, x):
        x = self.model._conv_stem(x)
        x = self.model._bn0(x)
        x = self.model._swish(x)
        feature_maps = []

        # TODO: temporarily storing extra tensor last_x and del it later might not be a good idea,
        #  try recording stride changing when creating efficientnet,
        #  and then apply it here.
        last_x = None
        for idx, block in enumerate(self.model._blocks):
            drop_connect_rate = self.model._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.model._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

            if block._depthwise_conv.stride == [2, 2]:
                feature_maps.append(last_x)
            elif idx == len(self.model._blocks) - 1:
                feature_maps.append(x)
            last_x = x
        del last_x
        return feature_maps[1:]

img_tfs = tfs.Compose([
    tfs.Resize((256,256)),
    tfs.ToTensor(),
    tfs.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
])


if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    model = EfficientNetFeatures(0).eval()
    writer = SummaryWriter('runs',comment = 'features')
    model.load_state_dict(torch.load('weights/efficientnet-b0.pth'),strict = False)
    if use_cuda:
        model = model.cuda()
    imgs = Images('./imgs', img_tfs)
    img_loader = DataLoader(imgs, 1, pin_memory = True)
    with torch.no_grad():
        for i, img in tqdm(enumerate(img_loader)):
            if use_cuda:
                img = img.cuda()
            features = model(img)
            for j in range(len(features)):
                fmin = features[j].min()
                fmax = features[j].max()
                fmean = features[j].mean()
                writer.add_scalars('features',{f'min_{j}':fmin,f'max_{j}':fmax,f'mean_{j}':fmean}, i)
            #writer.add_images('features', features, i)
    writer.close()
    



    
