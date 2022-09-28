import pandas as pd
import numpy as np
import torch
import sklearn.neighbors
import scipy.sparse as sp
from torch_geometric.data import Data


def sparse_dense_mul(s, d):
    i = s._indices()
    v = s._values()
    dv = d[i[0, :], i[1, :]]  # get values from relevant entries of dense matrix
    return torch.sparse.FloatTensor(i, v * dv, s.size())


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
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])),
                      shape=(ann_data.n_obs, ann_data.n_obs))
    G = G + sp.eye(G.shape[0])
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


def mclust_R(representation, n_clusters, random_seed=2022, model_name="EEE"):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as ro
    from rpy2.robjects import numpy2ri
    numpy2ri.activate()
    ro.r.library("mclust")
    r_random_seed = ro.r['set.seed']
    r_random_seed(random_seed)
    rmclust = ro.r['Mclust']
    res = rmclust(representation, n_clusters, model_name)
    mclust_res = np.array(res[-2])
    numpy2ri.deactivate()
    return mclust_res.astype('int')


def prepare_adata(adata):
    G_df = adata.uns['Spatial_Net'].copy()
    cells = np.array(adata.obs_names)
    cells_id_tran = dict(zip(cells, range(cells.shape[0])))
    G_df['Cell1'] = G_df['Cell1'].map(cells_id_tran)
    G_df['Cell2'] = G_df['Cell2'].map(cells_id_tran)
    G = sp.coo_matrix((np.ones(G_df.shape[0]), (G_df['Cell1'], G_df['Cell2'])), shape=(adata.n_obs, adata.n_obs))
    G = G + sp.eye(G.shape[0])
    edge_list = np.nonzero(G)
    data = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])),
                x=torch.FloatTensor(adata.X.todense()))  # .todense()
    return data
