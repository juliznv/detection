import os
import argparse
import yaml
import torch
import numpy as np

from tqdm.autonotebook import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms

from models import EfficientDet, FocalLoss
from datasets import CocoDataset, Resizer, Normalizer, Augmenter, collater,VIDDataset
from utils import init_weights

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

def get_args():
    parser = argparse.ArgumentParser('Yet Another EfficientDet Pytorch: SOTA object detection network - Zylo117')
    parser.add_argument('-p', '--project', type=str, default='coco', help='project file that contains parameters')
    parser.add_argument('-c', '--compound_coef', type=int, default=0, help='coefficients of efficientdet')
    parser.add_argument('-n', '--num_workers', type=int, default=1, help='num_workers of dataloader')
    parser.add_argument('--no-cuda', action = 'store_true', help = 'use no-cuda to ban cuda')
    parser.add_argument('--batch_size', type=int, default=12, help='The number of images per batch among all devices')
    parser.add_argument('--head_only', type=bool, default=False,
                        help='whether finetunes only the regressor and the classifier, '
                             'useful in early stage convergence or small/easy dataset')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optim', type=str, default='adamw', help='select optimizer for training, '
                                                                   'suggest using \'admaw\' until the'
                                                                   ' very final stage then switch to \'sgd\'')
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--val_interval', type=int, default=1, help='Number of epoches between valing phases')
    parser.add_argument('--save_interval', type=int, default=50, help='Number of steps between saving')
    parser.add_argument('--es_min_delta', type=float, default=0.0,
                        help='Early stopping\'s parameter: minimum change loss to qualify as an improvement')
    parser.add_argument('--es_patience', type=int, default=0,
                        help='Early stopping\'s parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.')
    parser.add_argument('--data_path', type=str, default='datasets/', help='the root folder of dataset')
    parser.add_argument('--log_path', type=str, default='logs/')
    parser.add_argument('-w', '--load_weights', type=str, default=None,
                        help='whether to load weights from a checkpoint, set None to initialize, set \'last\' to load last checkpoint')
    parser.add_argument('--saved_path', type=str, default='logs/')
    parser.add_argument('--debug', type=bool, default=False,
                        help='whether visualize the predicted boxes of training, '
                             'the output images will be in test/')

    args = parser.parse_args()
    return args


def save_checkpoint(model, name, opt):
    if not os.path.exists(opt.saved_path):
        os.mkdir(opt.saved_path)
    torch.save(model.state_dict(), os.path.join(opt.saved_path, name))

def train(dataloader, model, criterion, optimizer, opt):
    bset_loss = 1e10
    use_cuda = True if torch.cuda.is_available() and not opt.no_cuda else False
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        criterion = criterion.to('cpu')
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    for epoch in range(opt.num_epochs):
        epoch_loss = []
        progress_bar = tqdm(dataloader)
        for i, data in enumerate(progress_bar):
            img = data['img']
            target = data['annot']
            if use_cuda:
                img, target = img.cuda(), target.cuda()
            regression, classification, anchors = model(img)
            optimizer.zero_grad()
            cls_loss, reg_loss = criterion(classification, regression, anchors, target )
            loss = cls_loss.mean() + reg_loss.mean()
            if loss == 0 or not torch.isfinite(loss):
                continue
            loss.backward()
            epoch_loss.append(loss.item())
            progress_bar.set_description(
                'Epoch: {}/{}. Iteration: {}/{}. Cls loss: {:.5f}. Reg loss: {:.5f}. Total loss: {:.5f}'.format(
                    epoch + 1, opt.num_epochs, i + 1, len(dataloader), cls_loss.item(),
                    reg_loss.item(), loss.item()))
        
        #scheduler.step(np.mean(epoch_loss))
        if np.mean(epoch_loss) < bset_loss:
            bset_loss = np.mean(epoch_loss)
            print(f"Best loss:{bset_loss}, weights saved.")
            save_checkpoint(model, f'{opt.project}_efficientdet-d{opt.compound_coef}.pth',opt)


    

input_sizes = [512, 640, 768, 896, 1024, 512, 1280, 1536, 1536]

opt = get_args()
params = Params(f'configs/{opt.project}.yml')


if __name__ == '__main__':
    model = EfficientDet(num_classes = len(params.obj_list),compound_coef = opt.compound_coef,
                         ratios=eval(params.anchors_ratios), scales=eval(params.anchors_scales))
    criterion = FocalLoss()
    
    if opt.load_weights is not None:
        weights_path = opt.load_weights
        try:
            ret = model.load_state_dict(torch.load(weights_path), strict=False)
        except RuntimeError as e:
            print(f'[Warning] Ignoring {e}')
            print('''
                [Warning] Don\'t panic if you see this, this might be because 
                you load a pretrained weights with different number of classes.
                The rest of the weights should be loaded already.
                ''')
        print(f'[Info] loaded weights: {os.path.basename(weights_path)}')
    else:
        print('[Info] initializing weights...')
        init_weights(model)
    if opt.head_only:
        def freeze_backbone(m):
            classname = m.__class__.__name__
            for ntl in ['EfficientNet', 'BiFPN']:
                if ntl in classname:
                    for param in m.parameters():
                        param.requires_grad = False

        model.apply(freeze_backbone)
        print('[Info] freezed backbone')
    
    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), opt.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), opt.lr, momentum=0.9, nesterov=True)

    train_params = {'batch_size': opt.batch_size,
                    'shuffle': False,
                    'drop_last': False,
                    'collate_fn': collater,
                    'num_workers': opt.num_workers,
                    'pin_memory': True}
    '''
    trainset = CocoDataset(root_dir = os.path.join(opt.data_path,params.project_name),set = params.val_set,
                           transform=transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                                                         Augmenter(),
                                                         Resizer(input_sizes[opt.compound_coef])]))
    '''
    data_dir = os.path.join(opt.data_path, params.train_set)
    ann_dir = os.path.join(opt.data_path, 'annotations', params.train_set)
    tfs = transforms.Compose([Normalizer(mean=params.mean, std=params.std),
                              Augmenter(),
                              Resizer(input_sizes[opt.compound_coef])])
    for f in os.listdir(data_dir):
        img_dir = os.path.join(data_dir,f)
        ann_file = os.path.join(ann_dir,f + '.json')
        trainset = VIDDataset(img_dir,ann_file,tfs)
        trainloader = DataLoader(trainset,**train_params)                        
        train(trainloader,model,criterion,optimizer,opt)
