import seaborn as sns
import scanpy as sc
from scipy.spatial import distance
from scipy.cluster import hierarchy


def plot_consensus_map(cmat, method="average", return_linkage=True, **kwargs):
    row_linkage = hierarchy.linkage(distance.pdist(cmat), method=method)
    col_linkage = hierarchy.linkage(distance.pdist(cmat.T), method=method)
    figure = sns.clustermap(cmat, row_linkage=row_linkage, col_linkage=col_linkage, **kwargs)
    if return_linkage:
        return row_linkage, col_linkage, figure
    else:
        return figure


def plot_spatial(ann_data, ax, ):
    pass