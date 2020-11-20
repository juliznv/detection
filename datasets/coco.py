import os
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image

class COCODetect(Dataset):
    def __init__(self, data_dir, set_name = 'train2017',
                 transform = None,target_transform = None):
        super(COCODetect,self).__init__()
        self.data_dir =  data_dir
        self.set_name = set_name
        self.transform = transform
        self.target_transform = target_transform
        ann_file = os.path.join(data_dir, f'annotations/instances_{set_name}.json')
        self.coco = COCO(ann_file)
        self.img_ids = list(sorted(self.coco.imgs.keys()))
        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img_path = os.path.join(self.data_dir, self.set_name, img_path)
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img =self.transform(img)
        if 'train' in self.set_name:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target
        elif 'val' in self.set_name:
            metas = [self.coco.loadImgs(img_id)[0][i] for i in ['height','width']]
            return img, img_id, torch.Tensor(metas)
        else:
            return img

    def __len__(self):
        return len(self.img_ids)

if __name__ == '__main__':
    print("This is a Demo.")
    data = COCODetect(r'D:\data\coco', 'val2017')
    print(data[0])