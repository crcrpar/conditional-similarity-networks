import torch

from csn import ConditionalSimNet
import resnet_18
from tripletnet import CS_Tripletnet
from .tsne import BHTSNE


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


def get_mask(state_path, mask_cond=None, n_conditions=4, embedding_size=64):
    """Load trained mask(numpy.ndarray)"""
    csn = load_trained_csn(state_path, n_conditions, embedding_size)
    mask = csn.masks.weight.data.numpy()
    return mask


def embed_feature(feature, n_dim=2, perplexity=30, theta=.5, seed=-1):
    """Compress features into `n_dim`"""
    return BHTSNE(n_dim=n_dim, perplexity=perplexity, theta=theta, seed=seed).fit_transform(feature)
