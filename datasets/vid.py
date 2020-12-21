import os
import cv2
import numpy as np
from pycocotools.coco import COCO
from PIL import Image
import torch
from torch.utils.data import Dataset

class VID(Dataset):
    r'''Implement of dataset for ImageNet VID

    Arguments:
        root (str): ImageNet of root directory
        data_type (str):  Type of dataset train, val or test
        vid_file (str): Video directory, include all frames
        transform (callable): Convert image to appropriate format
        target_transform (callable): Convert target to appropriate formate

    '''
    def __init__(self, root, data_type, vid_file,
                 transform = None, target_transform = None):
        super(VID, self).__init__()
        ann_file = os.path.join(root,f"Annotations/{vid_file}.json")
        self.img_dir = os.path.join(root, 'Data\\VID', data_type, vid_file)
        self.coco = COCO(ann_file)
        self.transform = transform
        self.target_transform = target_transform
        self.img_ids = list(sorted(self.coco.imgs.keys()))
    
    def __getitem__(self, index):
        img_id = self.img_ids[index]    # image index
        # Load target by annotation indexes
        ann_ids = self.coco.getAnnIds(imgIds = img_id)
        target = self.coco.loadAnns(ann_ids)
        if self.target_transform:
            target = self.target_transform(target)
        # Load image by PIL.Image
        img_path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.img_dir,img_path)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        '''
        Return the length of current dataset
        '''
        return len(self.img_ids)

class VIDDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):

        self.img_dir = img_dir
        self.ann_file = ann_file
        self.transform = transform

        self.coco = COCO(ann_file)
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
        path = os.path.join(self.img_dir, image_info['file_name'])
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
    root = r'D:\data\ILSVRC'
    import torchvision.transforms as tfs
    ls = os.listdir(os.path.join(root,'Data\\VID\\val'))
    for i in ls:
        dataset = VID(root,'val',i,tfs.ToTensor())
        print(len(dataset[0][1]))


