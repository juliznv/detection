import os
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from PIL import Image


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
        
        
if __name__ == '__main__':
    root = r'D:\data\ILSVRC'
    import torchvision.transforms as tfs
    ls = os.listdir(os.path.join(root,'Data\\VID\\val'))
    for i in ls:
        dataset = VID(root,'val',i,tfs.ToTensor())
        print(len(dataset[0][1]))


