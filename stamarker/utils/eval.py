from typing import List
import numpy as np
import itertools
import torch
import tqdm
from sklearn.metrics import adjusted_rand_score
from scipy.cluster import hierarchy
from scipy.spatial import distance
import pandas as pd


def mclust_R(representation, n_clusters, r_seed=2022, model_name="EEE"):
    """
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(r_seed)
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    ro.r.library("mclust")
    r_random_seed = ro.r['set.seed']
    r_random_seed(r_seed)
    rmclust = ro.r['Mclust']
    res = rmclust(representation, n_clusters, model_name)
    mclust_res = np.array(res[-2])
    numpy2ri.deactivate()
    return mclust_res.astype('int')


def labels_connectivity_mat(labels: np.ndarray):
    _labels = labels - np.min(labels)
    n_classes = np.unique(_labels)
    mat = np.zeros([labels.size, labels.size])
    for i in n_classes:
        indices = np.squeeze(np.where(_labels == i))
        row_indices, col_indices = zip(*itertools.product(indices, indices))
        mat[row_indices, col_indices] = 1
    return mat


def consensus_matrix(labels_list: List[np.ndarray]):
    mat = 0
    for labels in labels_list:
        mat += labels_connectivity_mat(labels)
    return mat / float(len(labels_list))


def adjusted_rand_index(labels_true, labels_pred, warn=True):
    if isinstance(labels_true, pd.Series):
        indices = labels_true.isna()
    elif isinstance(labels_true, np.ndarray):
        indices = np.isnan(labels_true)
    else:
        raise TypeError
    n_nans = np.sum(indices)
    if np.sum(indices) > 0 and warn:
        print(f"Warning: true labels contain #{n_nans} NaN values")
    indices = np.logical_not(indices)
    return adjusted_rand_score(labels_true[indices], labels_pred[indices])


def consensus_clustering(labels_list, n_clusters):
    cmat = consensus_matrix(labels_list)
    row_linkage = hierarchy.linkage(distance.pdist(cmat), method="average")
    return hierarchy.cut_tree(row_linkage, n_clusters).squeeze()


def compute_consensus_ari(rng, mclust_labels_list, ground_truth, n_auto_econders=2, repeat=2, n_clusters=7):
    ari_list = []
    total_auto_encoders = len(mclust_labels_list)
    for dummy_i in tqdm.tqdm(range(repeat)):
        indices = rng.choice(total_auto_encoders, n_auto_econders, replace=False)
        labels_list = [mclust_labels_list[index] for index in indices]
        consensus_labels = consensus_clustering(labels_list, n_clusters)
        ari_list.append(adjusted_rand_index(ground_truth, consensus_labels))
    return ari_list


def class_proportions(target):
    n_classes = len(np.unique(target))
    props = np.array([np.sum(target == i) for i in range(n_classes)])
    return props / np.sum(props)


class Welford(object):
    """ Implements Welford's algorithm for computing a running mean
    and standard deviation. Modified from https://gist.github.com/alexalemi/2151722.
    """

    def __init__(self, lst=None):
        self.k = 0
        self.M = 0
        self.S = 0

        self.__call__(lst)

    def update(self, x):
        if x is None:
            return
        self.k += 1
        newM = self.M + (x - self.M) * 1. / self.k
        newS = self.S + (x - self.M) * (x - newM)
        self.M, self.S = newM, newS

    def consume(self, lst):
        lst = iter(lst)
        for x in lst:
            self.update(x)

    def __call__(self, x):
        if hasattr(x, "__iter__"):
            self.consume(x)
        else:
            self.update(x)

    @property
    def mean(self):
        return self.M

    @property
    def std(self):
        if self.k == 1:
            return 0
        if isinstance(self.S, torch.Tensor):
            return torch.sqrt(self.S / (self.k - 1))
        else:
            return np.sqrt(self.S / (self.k - 1))
