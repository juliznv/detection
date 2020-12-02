import os,shutil
import torch
import argparse

import torchvision.transforms as tfs

from torchvision.datasets import ImageFolder
from torchvision.models import *
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model import Classifier

parser = argparse.ArgumentParser(description='PyTorch Classifier Implement')
parser.add_argument('--data','-d',type=str,
    metavar='dir',help='path to dataset')
parser.add_argument('--model','-m',type=str,
    metavar='*.pt',help='pretrained model or save path')
parser.add_argument('--epochs','-e',type=int,default=10,
    help='num of epoches to run')
parser.add_argument('--batch-size','-bs',type=int,default=16,
    help='mini batch size (default : 16)')
parser.add_argument('--learning-rate','-lr',type=float,default=1e-3,
    help='initial learning rate (default : 1e-3)')
parser.add_argument('--momentum','-mm',type=float,default=0.9,
    help='optimizer momentum')
parser.add_argument('--weight-decay','-wd',type=float,default=1e-4,
    help='optimizer weight decay')
parser.add_argument('--cuda','-c',action='store_false',default=True,
    help='use cuda or not')

args = parser.parse_args()

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(loader,model,criterion,optimizer,epoch):
    if os.path.exists('./runs'):
        shutil.rmtree('./runs')
    writer=SummaryWriter(log_dir='./runs')
    model.train()
    use_cuda=args.cuda and torch.cuda.is_available()
    
    if use_cuda:
        print("Using CUDA to train.")
        model.cuda()
        criterion.cuda()

    for e in range(epoch):
        runing_loss=.0
        for i,(img,target) in enumerate(loader):
            if use_cuda:
                img ,target = img.cuda(), target.cuda()

            pred=model(img)
            loss=criterion(pred,target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            runing_loss+=loss.item()

            if i % 100 == 99:
                print("[%2d/%2d] %2.3f%%"%(e+1,epoch,(i+1)*100/len(loader)),
                    'loss: ',runing_loss/100)
                writer.add_scalar('loss',runing_loss/100,global_step=e*len(loader)+i)
                runing_loss=.0
                torch.save(model.state_dict(),args.model)

    writer.close()

def validate(loader,model,criterion):
    model.eval()
    use_cuda = args.cuda and torch.cuda.is_available()
    if use_cuda:
        model.cuda()
        criterion.cuda()
    tloss=0.0
    top1,top5=.0,.0
    with torch.no_grad():
        for i,(img,target) in enumerate(loader):
            if use_cuda:
                img, target = img.cuda(), target.cuda()
            
            pred=model(img)
            loss=criterion(pred,target)
            acc1, acc5 = accuracy(pred,target,(1,5))
            tloss+=loss.item()
            top1+=acc1
            top5+=acc5
    
    print('loss: %2.3f, top1: %2.3f, top5: %2.3f'%(tloss/len(loader),top1/len(loader),top5/len(loader)))
    
img_tfs=tfs.Compose([
    tfs.RandomResizedCrop(224),
    tfs.RandomHorizontalFlip(),
    tfs.ToTensor(),
    tfs.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    net = Classifier()
    if os.path.exists(args.model):
        net.load_state_dict(torch.load(args.model))
    
    imgnet = ImageFolder(args.data,transform=img_tfs)
    loader = DataLoader(imgnet,args.batch_size,shuffle=True,num_workers=6,pin_memory=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(net.parameters(),args.learning_rate)
    #optimizer=torch.optim.SGD(net.parameters(),args.learning_rate,args.momentum,weight_decay=args.weight_decay)
    
    '''
    '''
    train(loader,net,criterion,optimizer,args.epochs)
    validate(loader,net,criterion)




