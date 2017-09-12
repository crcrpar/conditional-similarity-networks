import argparse
from datetime import datetime as dt
import os

import numpy as np
import tqdm
import utils


def cluster_feature(feature_file_paths, n_components=2):
    root = os.path.dirname(feature_file_paths[0])
    out_paths = list()
    if n_components not in (2, 3):
        raise ValueError('invalid # of dimension')
    for path in tqdm.tqdm(feature_file_paths, desc='clustering'):
        feature_matrix = np.loadtxt(path, delimiter='\t')
        embedded_feature = utils.embedded_feature(feature_matrix)

        name, ext = os.path.splitext(os.path.basename(path))
        name += 'embedded_{}_dim'.format(n_components)
        filename = name + ext
        filepath = os.path.join(root, filename)
        np.savetxt(filepath, embedded_feature, delimiter='\t')
        out_paths.append(filepath)
    return out_paths
