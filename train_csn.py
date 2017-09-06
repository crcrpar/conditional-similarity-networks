from __future__ import print_function
import argparse
import os
import subprocess
from tqdm import tqdm

import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms as T

from csn import ConditionalSimNet
import resnet_18
from triplet_image_loader import TripletImageLoader
from tripletnet import CS_Tripletnet


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val, self.avg, self.sum, self.count = [.0] * 4

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_lr(args, optimizer, epoch, plotter=None):
    lr = args.lr * ((1 - 0.015) ** epoch)
    if args.visdom and plotter is not None:
        plotter.plot('lr', 'learning rate', epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(dista, distb):
    margin = .0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum() * 1.0 / dista.size(0)


def accuracy_id(dista, distb, c, c_id):
    margin = .0
    pred = (dista - distb - margin).cpu().data
    return ((pred > 0) * (c.cpu().data == c_id)).sum() * 1.0 / (c.cpu().data == c_id).sum()


def save_ckpt(args, state, is_best=True, filename='ckpt.pth'):
    root = os.path.join('runs, {}'.format(args.name))
    if not os.path.exists(root):
        os.makedirs(root)
    filename = os.path.join(root, filename)
    torch.save(state, filename)
    if is_best:
        src = filename
        dst = os.path.join(root, 'model_best.pth')
        cmd = 'cp {} {}'.format(src, dst)
        try:
            ret = subprocess.check_call(cmd.split(' '))
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                        help='number of start epoch (default: 1)')
    parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                        help='learning rate (default: 5e-5)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                        help='margin for triplet loss (default: 0.2)')
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--name', default='Conditional_Similarity_Network', type=str,
                        help='name of experiment')
    parser.add_argument('--embed_loss', type=float, default=5e-3, metavar='M',
                        help='parameter for loss for embedding norm')
    parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                        help='parameter for loss for mask norm')
    parser.add_argument('--num_traintriplets', type=int, default=100000, metavar='N',
                        help='how many unique training triplets (default: 100000)')
    parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                        help='how many dimensions in embedding (default: 64)')
    parser.add_argument('--test', dest='test', action='store_true',
                        help='To only run inference on test set')
    parser.add_argument('--learned', dest='learned', action='store_true',
                        help='To learn masks from random initialization')
    parser.add_argument('--prein', dest='prein', action='store_true',
                        help='To initialize masks to be disjoint')
    parser.add_argument('--conditions', nargs='*', type=int,
                        help='Set of similarity notions')
    parser.set_defaults(test=False)
    parser.set_defaults(learned=False)
    parser.set_defaults(prein=False)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    if args.conditions is not None:
        conditions = args.conditions
    else:
        conditions = list(range(4))

    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else dict()
    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json',
                           conditions, 'train', n_triplets=args.num_traintriplets,
                           transform=T.Compose([
                               T.Scale(112),
                               T.CenterCrop(112),
                               T.RandomHorizontalFlip(),
                               T.ToTensor(),
                               normalize,
                           ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json',
                           conditions, 'test', n_triplets=160000,
                           transform=T.Compose([
                               T.Scale(112),
                               T.CenterCrop(112),
                               T.ToTensor(),
                               normalize,
                           ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(
        TripletImageLoader('data', 'ut-zap50k-images', 'filenames.json',
                           conditions, 'val', n_triplets=80000,
                           transform=T.Compose([
                               T.Scale(112),
                               T.CenterCrop(112),
                               T.ToTensor(),
                               normalize,
                           ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)
    csn_model = ConditionalSimNet(model, n_conditions=len(conditions),
                                  embedding_size=args.dim_embed,
                                  learnedmask=args.learned, prein=args.prein)
    mask_var = csn_model.masks.weight
    triplet_net = CS_Tripletnet(csn_model)
    if args.cuda:
        triplet_net.cuda()

    if args.resume:
        is os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            triplet_net.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    parameters = filter(lambda p: p.requires_grad, triplet_net.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)
    n_param = sum([p.data.nelement() for p i triplet_net.parameters()])
    print('# of parameters: {}'.format(n_param))

    if args.test:
        test_acc = test(test_loader, triplet_net, args.epochs + 1)
        sys.exit()

    for epoch in tqdm(range(args.start_epoch, args.epochs + 1)):
        # TODO(crcrpar)
