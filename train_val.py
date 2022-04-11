# --------------------------------------------------------
# Cross-layer Non-Local Network
# Copyright (c) 2020 Zihan Ye
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import os
import argparse
import time
import shutil
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.parallel
from sklearn.metrics import f1_score,precision_score,recall_score
from torchvision import transforms
from termcolor import cprint
from torch.autograd import Variable
import torch.nn.functional as F
from lib import dataloader
from cnlnet_5CNL import model_hub
#from resnet import resnet50,resnet101
# torch version
cprint('=> Torch Vresion: ' + torch.__version__, 'green')

class FocalLoss(nn.Module):
    def __init__(self, gamma=2., reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        F_loss = ((1 - pt)**self.gamma) * CE_loss
        if self.reduction == 'sum':
            return F_loss.sum()
        elif self.reduction == 'mean':
            return F_loss.mean()

class FocalLosstrue(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.loss

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """
    def __init__(self, class_num=59, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        #print(class_mask)


        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        #print('probs size= {}'.format(probs.size()))
        #print(probs)

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        #print('-----bacth_loss------')
        #print(batch_loss)


        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss


# args
parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--debug', '-d', dest='debug', action='store_true',
        help='enable debug mode')
parser.add_argument('--warmup', '-w', dest='warmup', action='store_true',
        help='using warmup strategy')
parser.add_argument('--resume', '-r', dest='resume', action='store_true',
        help='using resume strategy')
parser.add_argument('--print-freq', '-p', default=1, type=int, metavar='N',
        help='print frequency (default: 10)')
parser.add_argument('--arch', default='50', type=str,
        help='the depth of resnet (default: 50)')
parser.add_argument('--valid', '-v', dest='valid',
        action='store_true', help='just run validation')
parser.add_argument('--checkpoints', default='', type=str,
        help='the dir of checkpoints')
parser.add_argument('--dataset', default='cub', type=str,
        help='cub | dog | car (default: cub)')
parser.add_argument('--lr', '--learning-rate', default=1e-5, type=float, metavar='LR',
        help='initial learning rate (default: 0.01)')

best_prec1 = 0
best_prec2 = 0

def main():
    global args
    global best_prec1, best_prec2

    args = parser.parse_args()

    # simple args
    debug = args.debug
    if debug: cprint('=> WARN: Debug Mode', 'yellow')

    dataset = args.dataset
    if dataset == 'cub':
        num_classes = 200
    elif dataset == 'dog':
        num_classes = 120
    elif dataset == 'car':
        num_classes = 196
    elif dataset == 'ai':
        num_classes = 59
    elif dataset == 'ip':
        num_classes = 102
    base_size = 580
    pool_size = 14 if base_size == 512 else 7
    workers = 0 if debug else 4
    batch_size = 4 if debug else 1
    if base_size == 512 and \
        args.arch == '152':
        batch_size = 128
    drop_ratio = 0.1
    if dataset == 'cub':
        lr_drop_epoch_list = [31, 51, 71]
    else:
        #lr_drop_epoch_list = [51, 71, 91]
        lr_drop_epoch_list = [16, 31, 46, 61, 76, 91]
    epochs = 100
    eval_freq = 1
    gpu_ids = [0] if debug else [0]
    #crop_size = 800
    log_name = "CNLtrainlog.txt"
    log_name2 = "CNLtrainlog2.txt"
    # args for the nl and cgnl block
    arch = args.arch

    # warmup setting
    WARMUP_LRS = [args.lr * (drop_ratio**len(lr_drop_epoch_list)), args.lr]
    WARMUP_EPOCHS = 0

    # data loader
    if dataset == 'cub':
        data_root = '/home/wqr/CUB_list'
        imgs_fold = os.path.join(data_root, 'images')
        train_ann_file = os.path.join(data_root, 'cub_train.list')
        valid_ann_file = os.path.join(data_root, 'cub_val.list')
    elif dataset == 'dog':
        data_root = '/input/data/Standford_dog'
        imgs_fold = os.path.join(data_root, 'images')
        train_ann_file = os.path.join(data_root, 'dog_train.list')
        valid_ann_file = os.path.join(data_root, 'dog_val.list')
    elif dataset == 'car':
        data_root = '/input/data/Standford_car'
        imgs_fold = os.path.join(data_root, 'images')
        train_ann_file = os.path.join(data_root, 'car_train.list')
        valid_ann_file = os.path.join(data_root, 'car_val.list')
    elif dataset == 'ai':
        data_root = 'aidemo'
        valid_root=os.path.join(data_root, 'ai')
        imgs_fold = os.path.join(data_root, 'ai')
        train_ann_file = os.path.join(data_root, 'train.list')
        valid_ann_file = os.path.join(data_root, 'train.list')
    elif dataset == 'ip':
        data_root = 'ipdemo'
        valid_root=os.path.join(data_root, 'ip')
        imgs_fold = os.path.join(data_root, 'ip')
        train_ann_file = os.path.join(data_root, 'train.list')
        valid_ann_file = os.path.join(data_root, 'train.list')
    else:
        raise NameError("WARN: The dataset '{}' is not supported yet.")

    train_dataset = dataloader.ImgLoader(
            root = imgs_fold,
            ann_file = train_ann_file,
            transform = transforms.Compose([
                #transforms.RandomResizedCrop(
                    #size=crop_size, scale=(0.08, 1.25)),
                transforms.Resize((base_size,base_size)),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(45),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean =[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225])
                ]))

    val_dataset = dataloader.ImgLoader(
            root = valid_root,
            ann_file = valid_ann_file,
            transform = transforms.Compose([
                transforms.Resize((base_size,base_size)),
                #transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225])
                ]))

    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size = batch_size,
            shuffle = True,
            num_workers = workers,
            pin_memory = True)

    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size = batch_size,
            shuffle = False,
            num_workers = workers,
            pin_memory = True)

    #build model
    model = model_hub(arch,
                     pretrained=False,pool_size=pool_size)
    
    #model = resnet50(pretrained=True)

    # change the fc layer
    model._modules['fc_m'] = torch.nn.Linear(in_features=2048,
                                           out_features=num_classes)
    torch.nn.init.kaiming_normal_(model._modules['fc_m'].weight,
                                  mode='fan_out', nonlinearity='relu')
    print(model)
    # parallel
    model = torch.nn.DataParallel(model, device_ids=gpu_ids).cuda()
    #criterion = FocalLoss(gamma=2).cuda()
    criterion = nn.CrossEntropyLoss().cuda()
    # optimizer
    optimizer = torch.optim.Adam(
            model.parameters(),
            args.lr,)
            #weight_decay=1e-4)
    #optimizer = torch.optim.SGD(
            #model.parameters(),
            #args.lr,
            #momentum=0.9,
            #weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 20,gamma=0.1)
    # cudnn
    cudnn.benchmark = True

    if args.resume:
            # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load('best_result.pth')
        best_acc = checkpoint['best_prec1']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # warmup
    if args.warmup:
        epochs += WARMUP_EPOCHS
        lr_drop_epoch_list = list(
                np.array(lr_drop_epoch_list) + WARMUP_EPOCHS)
        cprint('=> WARN: warmup is used in the first {} epochs'.format(
            WARMUP_EPOCHS), 'yellow')

    # valid
    if args.valid:
        cprint('=> WARN: Validation Mode', 'yellow')
        print('start validation ...')
        checkpoint_fold = args.checkpoints
        checkpoint_best = os.path.join(checkpoint_fold, 'best_result.pth')
        print('=> loading state_dict from {}'.format(checkpoint_best))
        #check = torch.load(checkpoint_best)
        #print('start aaaaaaaaaa ...',check)
        model.load_state_dict(
                torch.load(checkpoint_best)['state_dict'])
        #model.eval()
        prec1, prec2 = validate(val_loader, model, criterion, log_name)
        print(' * Final Accuracy: Prec@1 {:.3f}, Prec@2 {:.3f}'.format(prec1, prec2))
        exit(0)

    # train
    print('start training ...')
    
    f = open(log_name, 'w')
    f.write(log_name)
    f.close()
    
    for epoch in range(0, epochs):
        #current_lr = adjust_learning_rate(optimizer, drop_ratio, epoch, lr_drop_epoch_list,
                                          #WARMUP_EPOCHS, WARMUP_LRS)
        current_lr = 0
        # train one epoch
        train(train_loader, model, criterion, optimizer, epoch, epochs, current_lr,log_name2)
        scheduler.step()

        checkpoint_name = '{}-r-{}-w-CNL5-block.pth.tar'.format(dataset, arch)

        if (epoch + 1) % eval_freq == 0:
            prec1, prec2 = validate(val_loader, model, criterion, log_name)
            if prec1>best_prec1:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_prec1': prec1,
                    'optimizer' : optimizer.state_dict(),
                }, prec1, filename='best_result.pth')
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            best_prec2 = max(prec2, best_prec2)
            print(' * Best accuracy: Prec@1 {:.3f}, Prec@2 {:.3f}'.format(best_prec1, best_prec2))
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=checkpoint_name)


def train(train_loader, model, criterion, optimizer, epoch, epochs, current_lr,log_name2):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)
        # compute output
        output = model(input, save_attention=False)
        #output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output, target, topk=(1, 2))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1[0], input.size(0))
        top2.update(prec2[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0:3d}/{1:3d}][{2:3d}/{3:3d}]\t'
                  'LR: {lr:.7f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                   epoch, epochs, i, len(train_loader), 
                   lr=current_lr, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top2=top2))
    record = ' * Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f} loss{loss.avg:.4f}'.format(top1=top1, top2=top2,loss=losses)	
    f = open(log_name2, 'a')
    f.write(record+"\n")
    f.close()
    print(record)

    return top1.avg, top2.avg


def validate(val_loader, model, criterion, log_name):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top2 = AverageMeter()
    prediclist = []
    truelabel = []
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        save_attention = False
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda(non_blocking=True)

            input = input.cuda(non_blocking=True)
            # compute output
            output = model(input,save_attention)
            #output = model(input)
            prediction = torch.max(output, 1)[1]
            prediclist += prediction.cpu()
            truelabel += target.cpu()
            save_attention=False
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec2 = accuracy(output, target, topk=(1, 2))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top2.update(prec2[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@2 {top2.val:.3f} ({top2.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top2=top2))

        record = ' * Prec@1 {top1.avg:.3f} Prec@2 {top2.avg:.3f} loss{loss.avg:.4f}'.format(top1=top1, top2=top2,loss=losses)
        f = open(log_name, 'a')
        f.write(record+"\n")
        f.close()
        print(record)


        print('Weighted precision:', precision_score(truelabel, prediclist, average='weighted'))
        print('weighted recall:', recall_score(truelabel, prediclist, average='weighted'))
        print('weighted f1-score:', f1_score(truelabel, prediclist, average='weighted'))
        print('precision:', precision_score(truelabel, prediclist, average='macro'))
        print('recall:', recall_score(truelabel, prediclist, average='macro'))
        print('f1-score:', f1_score(truelabel, prediclist, average='macro'))

        w = open('f1.txt','a')
        w.write('Weighted precision:{}'.format(precision_score(truelabel, prediclist, average='weighted'))+"\n")
        w.write('Weighted recall:{}'.format(recall_score(truelabel, prediclist, average='weighted'))+"\n")
        w.write('Weighted f1-score:{}'.format(f1_score(truelabel, prediclist, average='weighted'))+"\n")
        w.write('precision:{}'.format(precision_score(truelabel, prediclist, average='macro'))+"\n")
        w.write('recall:{}'.format(recall_score(truelabel, prediclist, average='macro'))+"\n")
        w.write('f1-score:{}'.format(f1_score(truelabel, prediclist, average='macro'))+"\n")
        w.close()
    return top1.avg, top2.avg


def adjust_learning_rate(optimizer, drop_ratio, epoch, lr_drop_epoch_list,
                         WARMUP_EPOCHS, WARMUP_LRS):
    if args.warmup and epoch < WARMUP_EPOCHS:
        # achieve the warmup lr
        lrs = np.linspace(WARMUP_LRS[0], WARMUP_LRS[1], num=WARMUP_EPOCHS)
        cprint('=> warmup lrs {}'.format(lrs), 'green')
        for param_group in optimizer.param_groups:
            param_group['lr'] = lrs[epoch]
        current_lr = lrs[epoch]
    else:
        decay = drop_ratio if epoch in lr_drop_epoch_list else 1.0
        for param_group in optimizer.param_groups:
            param_group['lr'] = args.lr * decay
        args.lr *= decay
        current_lr = args.lr
    return current_lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()
