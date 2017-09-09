import argparse
from datetime import datetime as dt
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import plotly.plotly as py

from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import torch
from torch.autograd import Variable
import tqdm

from csn import ConditionalSimNet
import resnet_18
from tripletnet import CS_Tripletnet
import zappos_data


def load_trained_csn(state_path, n_conditions=4, embedding_size=64):
    """Load trained Conditional Similarity Network."""
    cnn = resnet_18.resnet18()
    csn = ConditionalSimNet(cnn, n_conditions, embedding_size)
    tnet = CS_Tripletnet(csn)
    ckpt = torch.load(state_path)
    tnet.load_state_dict(ckpt['state_dict'])
    tnet.eval()
    csn = tnet.embeddingnet
    return csn


def extract_feature(model, loader, cuda):
    for img_batch, c in tqdm.tqdm(loader, desc='feature extraction'):
        if cuda:
            img_batch = img_batch.cuda()
            c = c.cuda()
        x = Variable(img_batch, requires_grad=False, volatile=True)
        c = Variable(c, requires_grad=False, volatile=True)
        feature = model(x, c)
        if cuda:
            feature = feature.cpu()
        feature_np = feature.data.numpy()
        yield feature_np


def dump_feature(model, loader, condition, path, cuda):
    with open(path, 'a') as f:
        for feature in extract_feature(model, loader, condition, cuda):
            np.savetxt(f, feature, delimiter='\t')


def dump_feature_files(root, base_path, files_json_path, batch_size,
                       conditions, out_dir, out_file, state_path, cuda):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # load trained CSN
    print("... loading {}\n".format(state_path))
    trained_csn = load_trained_csn(state_path)
    if cuda:
        trained_csn.cuda()

    feature_files_dict = dict()
    for condition in tqdm.tqdm(conditions, desc='condition'):
        # path
        path = os.path.join(out_dir, out_file.format(condition))
        feature_files_dict[condition] = path
        # prepare a loader
        kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}
        loader = zappos_data.make_data_loader(
            condition, root, base_path, files_json_path, batch_size, **kwargs)
        # start extracting features
        start_time = dt.now()
        path = os.path.join(out_dir, out_file.format(condition))
        dump_feature(trained_csn, loader, condition, path, cuda)
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
    embedded_feature = tsne.fit_transform(feature_matrix)

    root = os.path.dirname(feature_file_path)
    filename = os.path.basename(feature_file_path)
    name, ext = os.path.splitext(filename)
    name += 'embedded'
    filename = name + ext
    filepath = os.path.join(root, filename)
    np.savetxt(filepath, embedded_feature, delimiter='\t')


def scatter(embedded_feature, out_dir, condition, files=None, plotly=False):
    categories = [f.split('/')[:2] for f in files]
    if files is not None:
        category_ids = LabelEncoder().fit(categories).fit_transform(categories)
        category_list = list(set(category_ids))
    fig, ax = plt.subplots()
    x, y = np.hsplit(embedded_feature, 1)
    ax.scatter(x, y, alpha=.3)
    plt.savefig(os.path.join(out_dir, '{}.png'.format(condition)))

    if plotly:
        plot_url = py.plot_mpl(fig, filename="mpl-scatter")
        return plot_url
    else:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='data',
                        help='path to directory including ut-zap50k-images')
    parser.add_argument('--base_path', default='ut-zap50k-images',
                        help='directory name of ut-zap50k-images')
    parser.add_argument('--files_json_path', default='filenames.json',
                        help='json file name which contains all the relative paths to images')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--conditions', default=None,
                        help='condition to visualize. default is all')
    parser.add_argument('--out_dir', default='visualization',
                        help='directory to save features')
    parser.add_argument('--out_file', default='condition_{}.tsv',
                        help='file to save features')
    parser.add_argument('--state_path',
                        default='runs/Conditional_Similarity_Network/model_best.pth.tar')
    parser.add_argument('--n_components', default=2, type=int,
                        help='the number of dimensions to execute t-SNE')
    parser.add_argument('--cuda', default=1,
                        help='0 indicates CPU mode')
    parser.add_argument('--debug', default=0,
                        help='1 indicates debug')
    args = parser.parse_args()

    # extract feature
    if args.conditions is None:
        conditions = list(range(4))
    else:
        assert args.conditions in [0, 1, 2, 3]
        conditions = list(args.conditions)

    if args.debug:
        import sys
        print('###=== DEBUG ===###')
        print('# check dataset')
        dataset = zappos_data.make_dataset(args.root,
                                           args.base_path,
                                           args.files_json_path)
        print('## len(dataset) = {}'.format(len(dataset)))
        print('# check loader')
        loader = torch.utils.data.DataLoader(dataset=dataset,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=4,
                                             pin_memory=True)
        print('## len(loader) = {}'.format(len(loader)))
        print('## loader.sampler = {}'.format(type(loader.sampler)))
        print('## run iteration')
        for idx, batch in enumerate(loader):
            if idx % 50 == 0:
                print('{}th batch'.format(idx))
                x = Variable(batch, requires_grad=False, volatile=True)
                print('x.requires_grad {}, x.volatile {}'.format(
                    x.requires_grad, x.volatile))
        sys.exit()

    feature_files_dict = dump_feature_files(
        args.root, args.base_path, args.files_json_path, args.batch_size,
        conditions, args.out_dir, args.out_file, args.state_path, args.cuda)

    # compress features to {args.n_components}-D
    for condition in tqdm.tqdm(conditions, desc='t-SNE'):
        cluster_feature(feature_files_dict, condition, args.n_components)


if __name__ == '__main__':
    main()
