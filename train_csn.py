import argparse
from datetime import datetime as dt
import os
from six.moves import cPickle as pickle
import subprocess
import tqdm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms as T

from csn import ConditionalSimNet
import resnet_18
from triplet_image_loader import TripletImageLoader
from tripletnet import CS_Tripletnet


class AverageMeter(object):

    def __init__(self):
        self.val = .0
        self.avg = .0
        self.sum = .0
        self.count = 0

    def reset(self):
        self.val, self.avg, self.sum, self.count = .0, .0, .0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Trainer(object):

    def __init__(self, model, optimizer, accuracy, criterion, train_loader,
                 val_loader, cuda):
        self.model = model
        self.optimizer = optimizer
        self.accuracy = accuracy
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cuda = cuda
        self.log = dict()

    def run(self, n_epoch, log_interval, ckpt_interval, best_only=True):
        for epoch in tqdm.tqdm(range(n_epoch), desc='tota'):
            self.log['epoch{}'.format(epoch)] = {
                'train': dict(), 'validate': dict()}
            self.train(eopch, log_interval)
            self.validate(epoch)

    def train(self, log_interval):
        model = self.model
        optimizer = self.optimizer
        accuracy = self.accuracy
        criterion = self.criterion
        train_loader = self.train_loader

        model.train()
        for batch_idx, (data1, data2, data3, c) in tqdm.tqdm(enumerate(train_loader), desc='training'):
            if self.cuda:
                data1, data2, data3 = data1.cuda(), data2.cuda(), data3.cuda()
                c = c.cuda()
            data1, data2, data3, c = Variable(data1), Variable(
                data2), Variable(data3), Variable(c)

            dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm = model(
                data1, data2, data3, c)
            target = torch.FloatTensor(dist_a.size()).fill_(1)
            if self.cuda:
                target = target.cuda()
            target = Variable(target)

            loss_triplet = criterion(dist_a, dist_b, target)
            loss_embedd = embed_norm / np.sqrt(data1.size(0))
            loss_mask = mask_norm / data1.size(0)
            loss = loss_triplet + self.lam_embed * loss_embedd + self.lam_mask * loss_mask
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = accuracy(dist_a, dist_b)
            size = data1.size(0)
            current_log = {
                'loss': loss_triplet.data[0] / size,
                'acc': acc / size,
                'embed_norm': loss_embedd.data[0],
                'mask_norm': loss_mask.data[0]
            }
            self.log['epoch{}'.format(epoch)]['batch{}'.format(
                batch_idx)] = current_log

            if batch_idx % log_interval == 0:
                tqdm.tqdm.write('loss: {:.4f}\tacc: {:.4f}\tembedded_norm: {:.2f}'.format(
                    current_log['loss'], current_log['acc'], current_log['embed_norm'], current_log['mask_norm']))

    def validate(self):
        model = self.model
        accuracy = self.accuracy
        criterion = self.criterion
        val_loader = self.val_loader

        losses, accs, acc_cond = list(), list(), dict()
        for condition in conditions:
            acc_cond[condition] = list()

        model.eval()
        for batch_idx, (data1, data2, data3, c) in tqdm.tqdm(enumerate(val_loader), desc='validation'):
            if self.cuda:
                data1, data2, data3, c = data1.cuda(), data2.cuda(), data3.cuda(), c.cuda()
            data1, data2, data3, c = Variable(data1), Variable(
                data2), Variable(data3), Variable(c)
            c_test = c

            dist_a, dist_b, _, _, _ = model(data1, data2, data3, c)
            target = torch.FloatTensor(dist_a.size()).fill_(1)
            if self.cuda:
                target = target.cuda()
            target = Variable(target)
            test_loss = criterion(dist_a, dist_b, target).data[0]
            losses.append(test_loss)
            acc = accuracy(dist_a, dist_b)
            accs.append(acc)
            for condition in conditions:
                acc_cond[condition].append(accuracy_id(
                    dist_a, dist_b, c_test, condition))

        self.log['epoch{}'.format(epoch)]['test'] = {
            'loss': losses / len(val_loader),
            'acc': sum(accs) / len(val_loader)}
        for condition in conditions:
            self.log['epoch{}'.format(epoch)]['test']['{}'.format(
                condition)] = sum(acc_cond[condition]) / len(val_loader)


def adjust_lr(args, optimizer, epoch, plotter=None):
    lr = args.lr * ((1 - 0.015) ** epoch)
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


def save_ckpt(root, is_best=True, filename='ckpt.pth'):
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


def train(args, train_loader, triplet_net, criterion, optimizer, epoch):
    triplet_net.train()
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()
    mask_norms = AverageMeter()
    loss_acc_log = {'loss': list(), 'acc': list()}

    for batch_idx, (data1, data2, data3, c) in tqdm(enumerate(train_loader), desc='training loop'):
        if args.cuda:
            data1 = data1.cuda()
            data2 = data2.cuda()
            data3 = data3.cuda()
            c = c.cuda()
        data1 = Variable(data1)
        data2 = Variable(data2)
        data3 = Variable(data3)
        c = Variable(c)

        dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm = triplet_net(
            data1, data2, data3, c)
        target = torch.FloatTensor(dist_a.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)

        loss_triplet = criterion(dist_a, dist_b, target)
        loss_embedd = embed_norm / np.sqrt(data1.size(0))
        loss_mask = mask_norm / data1.size(0)
        loss = loss_triplet + args.embed_loss * loss_embedd +
            args.mask_loss * loss_mask

        losses.update(loss_triplet.data[0], data1.size(0))
        acc = accuracy(dist_a, dist_b)
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.data[0])
        mask_norms.update(loss_mask.data[0])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_acc_log['loss'].append(losses.val)
        loss_acc_log['acc'].append(accs.val)

        if batch_idx * args.log_interval == 0:
            tqdm.write('Epoch: {} [{}/{}]\t'
                       'Loss: {:.4f} ({:.4f})\t'
                       'Acc: {:,2f}% ({:.3f}%)\t'
                       'emb_norm: {:.2f} ({:.2f})'.format(
                           epoch, batch_idx * len(data1),
                           len(train_loader.dataset), losses.val, losses.avg,
                           100. * accs.val, 100. * accs.avg, emb_norms.val,
                           emb_norms.avg))
        return loss_acc_log


def test(args, conditions, test_loader, triplet_net, criterion, epoch):
    triplet_net.eval()
    losses = AverageMeter()
    accs = AverageMeter()
    accs_cs = dict([(key, AverageMeter()) for key in conditions])

    for batch_idx, (data1, data2, data3, c) in tqdm(enumerate(test_loader)):
        if args.cuda:
            data1 = data1.cuda()
            data2 = data2.cuda()
            data3 = data3.cuda()
            c = c.cuda()
        data1 = Variable(data1)
        data2 = Variable(data2)
        data3 = Variable(data3)
        c = Variable(c)
        c_test = c

        dist_a, dist_b, _, _ = triplet_net(data1, data2, data3, c)
        target = torch.FloatTensor(dist_a.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)
        test_loss = criterion(dist_a, dist_b, target).data[0]

        acc = accuracy(dist_a, dist_b)
        accs.update(acc, data1.size(0))
        for condition in conditions:
            accs_cs[condition].update(accuracy_id(
                dist_a, dist_b, c_test, condition), data1.size(0))
        losses.update(test_loss, data1.size(0))

    return losses, accs


def main():
    parser = argparse.ArgumentParser(description='This is a WIP program')
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
    parser.add_argument('--out', type=str, default='result',
                        help='dir to save models and log')
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
        if os.path.isfile(args.resume):
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
    n_param = sum([p.data.nelement() for p in triplet_net.parameters()])
    print('# of parameters: {}'.format(n_param))
    print('\n\n')

    if args.test:
        import sys
        test_loss, test_acc = test(test_loader, triplet_net, args.epochs + 1)
        print('accuracy: {}, loss: {}'.format(test_acc.avg, test_loss.avg))
        sys.exit()

    root = os.path.join(args.out, dt.now().strftime('%m%d_%H%M'))
    if not os.path.exists(root):
        os.makedirs(root)
    best_acc = .0
    log = dict()
    start_time = dt.now()
    for epoch in tqdm(range(args.start_epoch, args.epochs + 1), desc='total'):
        adjust_lr(args, optimizer, epoch)
        loss_acc_log = train(args, train_loader, triplet_net,
                             criterion, optimizer, epoch)
        log['epoch_{}_train'.format(epoch)] = loss_acc_log
        losses, accs = test(args, val_loader, triplet_net, criterion, epoch)
        log['epoch_{}_val'.format(epoch)] = {
            'loss': losses.avg, 'acc': 100. * accs.avg}
        tqdm.write('[validation]\nloss: {:.4f}\tacc: {:.2f}%\n'.format(
            losses.avg, 100. * accs.avg))

        is_best = accs.avg > best_acc
        best_acc = max(accs.avg, best_acc)
        save_ckpt(root, triplet_net.state_dict(), is_best)

    end_time = dt.now()
    print('\ntraining finished.')
    print('started at {}, ended at {}, duration is {} hours'.format(
        start_time.strftime('%m%d, %H:%M'), end_time.strftime('%m%d, %H:%M'),
        (end_time - start_time).total_seconds() / 3600.))
    save_ckpt(root, triplet_net.state_dict(), filename='model.pth')
    log_filepath = os.path.join(root, 'log.pkl')
    with open(log_filepath, 'wb') as f:
        pickle.dump(log, f)
    print('log files saved at {}'.format(log_filepath))


if __name__ == '__main__':
    main()
