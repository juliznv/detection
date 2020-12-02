import os
<<<<<<< HEAD
import torch
from torch.utils.data import Dataset
=======
import cv2
import numpy as np
>>>>>>> 13481d5fb855e05403b111c87a52328f59cf0bca
from pycocotools.coco import COCO
from PIL import Image
import torch
from torch.utils.data import Dataset,BatchSampler

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
<<<<<<< HEAD
            return img, target
        elif 'val' in self.set_name:
            metas = [self.coco.loadImgs(img_id)[0][i] for i in ['height','width']]
            return img, img_id, torch.Tensor(metas)
=======
            else:
                target = COCODetect.load_annotations(target)
            return img, target
        elif 'val' in self.set_name:
            metas = [self.coco.loadImgs(img_id)[0][i] for i in ['height','width']]
            return img, img_id, torch.tensor(metas)
>>>>>>> 13481d5fb855e05403b111c87a52328f59cf0bca
        else:
            return img

    def __len__(self):
        return len(self.img_ids)
    @staticmethod
    def load_annotations(anns):
        target = []
        for ann in anns:
            target += [ann['bbox'] + [ann['category_id']]]
        target = torch.tensor(target).to(torch.float32)
        if target.size(0) == 0:
            return target
        target[:,[2,3]] += target[:,[0,1]]
        return target


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
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

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations

if __name__ == '__main__':
    print("This is a Demo.")
    data = COCODetect(r'D:\data\coco', 'val2017')
    print(data[0])