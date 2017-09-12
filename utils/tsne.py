r"""t-SNE implementation from iwiwi blog post.

As you know, scikit-learn TSNE is slow.
see, http://iwiwi.hatenadiary.jp/?page=1477023358
"""
import bhtsne
import numpy as np
import sklearn.base


class BHTSNE(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):

    def __init__(self, n_dim=2, perplexity=30.0, theta=0.5, seed=-1):
        self.n_dim = n_dim
        self.perplexity = perplexity
        self.theta = theta
        self.seed = seed

    def fit_transform(self, x):
        tsne = bhtsne.tsne(x.astype(np.float64),
                           dimensions=self.dimensions,
                           perplexity=self.perplexity,
                           theta=self.theta,
                           seed=self.seed)
        return tsne
