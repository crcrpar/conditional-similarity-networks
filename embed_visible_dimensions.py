import os
import numpy as np
from tsne import BHTSNE


def _embed_feature_vector(features_path,
                          n_dim=2,
                          perplexity=30,
                          theta=0.5,
                          seed=-1):
    if n_dim not in (2, 3):
        raise ValueError('invalid dimensions')

    feature = np.loadtxt(features_path)
    embeded_feature = BHTSNE(n_dim=n_dim,
                             perplexity=perplexity,
                             theta=theta,
                             rand_seed=seed).fit_transform(feature)
    return embeded_feature


def embed_features(features_path, out_path=None, n_dim=2, perplexity=30,
                   theta=0.5, seed=-1):
    embeded_feature = _embed_feature_vector(features_path, n_dim, perplexity,
                                            theta, seed)
    if not os.path.isdir(os.path.basename(out_path)):
        os.makedirs(os.path.basename(out_path))
    np.savetxt(out_path, embeded_feature, delimiter='\t')
