import time
import yaml
import os
import seaborn as sns
import numpy as np
import pandas as pd
import scanpy as sc
import itertools
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import distance
from scipy.cluster import hierarchy
from scipy.optimize import linear_sum_assignment
import sklearn.neighbors
from typing import List


def plot_clustered_consensus_matrix(cmat, n_clusters, method="average", resolution=0.5,
                                    figsize=(5, 5)):
    n_samples = cmat.shape[0]
    linkage_matrix = hierarchy.linkage(cmat, method='average', metric='euclidean')
    cluster_labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
    visualization_clusters = hierarchy.fcluster(linkage_matrix, int(n_samples * resolution), criterion='maxclust')
    sorted_indices = np.argsort(visualization_clusters)
    sorted_cmat = cmat[sorted_indices][:, sorted_indices]
    figure, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(sorted_cmat, cmap='magma', interpolation='nearest')
    return figure, cluster_labels


class Timer:

    def __init__(self):
        self.timer_dict = {}
        self.stop_dict = {}

    def tic(self, name):
        self.timer_dict[name] = time.time()

    def toc(self, name):
        assert name in self.timer_dict
        elapsed = time.time() - self.timer_dict[name]
        del self.timer_dict[name]
        return elapsed

    def stop(self, name):
        self.stop_dict[name] = time.time()

    def resume(self, name):
        if name not in self.timer_dict:
            del self.stop_dict[name]
            return
        elapsed = time.time() - self.stop_dict[name]
        self.timer_dict[name] = self.timer_dict[name] + elapsed
        del self.stop_dict[name]


def save_yaml(yaml_object, file_path):
    with open(file_path, 'w') as yaml_file:
        yaml.dump(yaml_object, yaml_file, default_flow_style=False)

    print(f'Saving yaml: {file_path}')
    return


def parse_args(yaml_file):
    with open(yaml_file, 'r') as stream:
        try:
            cfg = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return cfg


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


# `consensus_matrix` is replaced by `compute_consensus_matrix` which is much more efficient by using linear sum assignment.
def consensus_matrix(labels_list: List[np.ndarray]):
    mat = 0
    for labels in labels_list:
        mat += labels_connectivity_mat(labels)
    return mat / float(len(labels_list))


def compute_consensus_matrix(clustering_results):
    """
    Compute the consensus matrix from M times clustering results.
    Parameters:
    -- clustering_results: numpy array of shape (M, n)
        M times clustering results, where M is the number of times clustering was performed
        and n is the number of data points or elements in the clustering results.
    Returns:
    -- consensus_matrix: numpy array of shape (n, n)
        Consensus matrix, where n is the number of data points or elements in the clustering results.
    """
    M, n = clustering_results.shape
    # Compute dissimilarity matrix between clustering results using cdist
    dissimilarity_matrix = distance.cdist(clustering_results, clustering_results, metric='hamming')
    # Compute consensus matrix using linear sum assignment
    row_ind, col_ind = linear_sum_assignment(dissimilarity_matrix)
    consensus_matrix = np.zeros((n, n))
    for i, j in zip(row_ind, col_ind):
        consensus_matrix += (clustering_results[i][:, np.newaxis] == clustering_results[j])
    # Divide the consensus matrix by the number of comparisons to obtain the consensus percentages
    consensus_matrix /= M
    return consensus_matrix


def compute_spatial_net(ann_data, rad_cutoff=None, k_cutoff=None,
                        max_neigh=50, model='Radius', verbose=True):
    """
    Construct the spatial neighbor networks.

    Parameters
    ----------
    ann_data
        AnnData object of scanpy package.
    rad_cutoff
        radius cutoff when model='Radius'
    k_cutoff
        The number of nearest neighbors when model='KNN'
    model
        The network construction model. When model=='Radius', the spot is connected to spots whose distance is less than rad_cutoff. When model=='KNN', the spot is connected to its first k_cutoff nearest neighbors.

    Returns
    -------
    The spatial networks are saved in adata.uns['Spatial_Net']
    """

    assert (model in ['Radius', 'KNN'])
    if verbose:
        print('------Calculating spatial graph...')
    coor = pd.DataFrame(ann_data.obsm['spatial'])
    coor.index = ann_data.obs.index
    coor.columns = ['imagerow', 'imagecol']

    nbrs = sklearn.neighbors.NearestNeighbors(
        n_neighbors=max_neigh + 1, algorithm='ball_tree').fit(coor)
    distances, indices = nbrs.kneighbors(coor)
    if model == 'KNN':
        indices = indices[:, 1:k_cutoff + 1]
        distances = distances[:, 1:k_cutoff + 1]
    if model == 'Radius':
        indices = indices[:, 1:]
        distances = distances[:, 1:]
    KNN_list = []
    for it in range(indices.shape[0]):
        KNN_list.append(pd.DataFrame(zip([it] * indices.shape[1], indices[it, :], distances[it, :])))
    KNN_df = pd.concat(KNN_list)
    KNN_df.columns = ['Cell1', 'Cell2', 'Distance']
    Spatial_Net = KNN_df.copy()
    if model == 'Radius':
        Spatial_Net = KNN_df.loc[KNN_df['Distance'] < rad_cutoff,]
    id_cell_trans = dict(zip(range(coor.shape[0]), np.array(coor.index), ))
    cell1, cell2 = Spatial_Net['Cell1'].map(id_cell_trans), Spatial_Net['Cell2'].map(id_cell_trans)
    Spatial_Net = Spatial_Net.assign(Cell1=cell1, Cell2=cell2)
    # Spatial_Net.assign(Cell1=Spatial_Net['Cell1'].map(id_cell_trans))
    # Spatial_Net.assign(Cell2=Spatial_Net['Cell2'].map(id_cell_trans))
    if verbose:
        print('The graph contains %d edges, %d cells.' % (Spatial_Net.shape[0], ann_data.n_obs))
        print('%.4f neighbors per cell on average.' % (Spatial_Net.shape[0] / ann_data.n_obs))
    ann_data.uns['Spatial_Net'] = Spatial_Net


def compute_edge_list(ann_data):
    G_df = ann_data.uns['Spatial_Net'].copy()
    cells = np.array(ann_data.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = scipy.sparse.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])),
                      shape=(ann_data.n_obs, ann_data.n_obs))
    G = G + scipy.sparse.eye(G.shape[0])
    edge_list = np.nonzero(G)
    return edge_list


def stats_spatial_net(ann_data):
    import matplotlib.pyplot as plt
    Num_edge = ann_data.uns['Spatial_Net']['Cell1'].shape[0]
    Mean_edge = Num_edge / ann_data.shape[0]
    plot_df = pd.value_counts(pd.value_counts(ann_data.uns['Spatial_Net']['Cell1']))
    plot_df = plot_df / ann_data.shape[0]
    fig, ax = plt.subplots(figsize=[3, 2])
    plt.ylabel('Percentage')
    plt.xlabel('')
    plt.title('Number of Neighbors (Mean=%.2f)' % Mean_edge)
    ax.bar(plot_df.index, plot_df)

def select_svgs(smap, domain_id, domain_labels, alpha=1.5):
    """
    Select spatial domain SVGs (spatially variable genes)
    """
    scores = np.linalg.norm(smap.iloc[domain_labels==domain_id, :], axis=0)
    mu, std = np.mean(scores), np.std(scores)
    return smap.columns[scores > mu + alpha * std].tolist()
