import argparse
from datetime import datetime as dt
import itertools
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import plotly.plotly as py

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import tqdm

from csn import ConditionalSimNet
import resnet_18
from tripletnet import CS_Tripletnet
import zappos_data


def load_trained_csn(state_path):
    cnn = resnet_18.resnet18()
    csn = ConditionalSimNet(cnn)
    tnet = CS_Tripletnet(csn)
    ckpt = torch.load(state_path)
    tnet.load_state_dict(ckpt['state_dict'])
    tnet.eval()
    csn = tnet.embeddingnet
    return csn


def extract_feature(model, loader, condition, cuda):
    if condition not in set([0, 1, 2, 3]):
        raise ValueError('invalid condition for zappos')

    for batch_idx, (batch_paths, img_batch) in tqdm.tqdm(enumerate(loader), desc='feature extraction'):
        if cuda:
            img_batch = img_batch.cuda()
        x = Variable(img_batch)
        feature = model(x)
        if cuda:
            feature = feature.cpu()
        feature_np = feature.data.numpy()
        yield feature_np


def dump_feature(model, loader, condition, cuda, path):
    with open(file_path, 'a') as f:
        for feature in extract_feature(model, loader, condition, cuda):
            np.savetxt(f, feature, delimiter='\t')


def calc_dump_feature_files(root, base_path, files_json_path, batch_size,
                            out_dir, out_file, state_path):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    trained_csn = load_trained_csn(state_path)
    feature_files_dict = dict()
    for split in tqdm.tqdm(all_split, desc='split'):
        feature_files_dict[split] = dict()
        _root = os.path.join(out_dir, split)
        if not os.path.exists(_root):
            os.makedirs(_root)
        for condition in tqdm.tqdm(conditions, desc='condition'):
            path = os.path.join(_root, out_file.format(condition=condition))
            feature_files_dict[split][condition] = path
            start_time = dt.now()
            loader = zappos_data.make_data_loader(
                root, base_path, files_json_path, path)
            dump_feature(model, loader, condition, cuda)
            end_time = dt.now()
            duration = (end_time - start_time).total_seconds() / 60.0
            tqdm.tqdm.write('duration: {:.2f}[min]'.format(duration))
            tqdm.tqdm.write('dump {}'.format(os.path.basename(path)))

    return feature_files_dict


def cluster_feature(feature_files_dict, split, condition, n_components=2):
    if n_components not in (2, 3):
        raise ValueError('invalid # of dimension')
    feature_file_path = feature_files_dict[split][condition]
    feature_matrix = np.load(feature_file_path)
    tsne = TSNE(n_components=n_components)
    tsne.fit(feature_matrix)
    embedded_feature = test.fit_transform(feature_matrix)

    root = os.path.dirname(feature_file_path)
    filename = os.path.basename(feature_file_path)
    name, ext = os.path.splitext(filename)
    name += 'embedded'
    filename = name + ext
    filepath = os.path.join(root, filename)
    np.savetxt(filepath, embedded_feature, delimiter='\t')


def scatter(embedded_feature, out_dir, split, condition, files=None, plotly=False):
    categories = [f.split('/')[:2] for f in files]
    if files is not None:
        category_ids = LabelEncoder().fit(categories).fit_transform(categories)
        category_list = list(set(category_ids))
    fig, ax = plt.subplots()
    x, y = np.hsplit(embedded_feature, 1)
    ax.scatter(x, y, alpha=.3)
    plt.savefig(os.path.join(out_dir, '{}_{}.png'.format(split, condition)))

    if plotly:
        plot_url = py.plot_mpl(fig, filename="mpl-scatter")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data',
                        help='path to directory including ut-zap50k-images')
    parser.add_argument('--base_path', default='ut-zap50k-images',
                        help='directory name of ut-zap50k-images')
    parser.add_argument('--files_json_path', default='filenames.json',
                        help='json file name which contains all the relative paths to images')
    parser.add_argument('--split', default=None,
                        help='dataset to visualize. default is all')
    parser.add_argument('--conditions', default=None,
                        help='condition to visualize. default is all')
    parser.add_argument('--out_dir', default='visualization',
                        help='directory to save features')
    parser.add_argument(
        '--state_path', default='runs/Conditional_Similarity_Network_bs_64/model_best.pth.tar')
    parser.add_argument('--n_components', default=2,
                        help='the number of dimensions to execute t-SNE')
    args = parser.parse_args()

    assert args.split in ['train', 'val', 'test']
    assert args.conditions in [0, 1, 2, 3]
    # extract feature
    if args.split is None:
        splits = ['train', 'val', 'test']
    else:
        splits = list(args.split)
    if args.conditions is None:
        conditions = list(range(4))
    else:
        conditions = list(args.conditions)
    feature_files_dict = calc_dump_feature_files(
        args.root, args.base_path, args.files_json_path, args.batch_size, args.out_dir, args.state_path)

    # compress features to {args.n_components}-D
    for (split, condition) in tqdm.tqdm(itertools.product(splits, conditions), desc='t-SNE'):
        cluster_feature(feature_files_dict, split,
                        condition, args.n_components)


if __name__ == '__main__':
    main()
