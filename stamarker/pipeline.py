import pytorch_lightning as pl
import copy
import torch
import torch.nn.functional as F
import os
import shutil
import logging
import glob
import sys
import numpy as np
import pandas as pd
import scipy
import scanpy as sc
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.cluster import hierarchy
from .models import intSTAGATE, StackClassifier
from .utils import plot_clustered_consensus_matrix, compute_consensus_matrix, Timer
from .dataset import SpatialDataModule
from .modules import STAGATEClsModule
import logging

FORMAT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')


def make_spatial_data(ann_data):
    """
    Make SpatialDataModule object from Scanpy annData object
    """
    data_module = SpatialDataModule()
    ann_data.X = ann_data.X.toarray()
    data_module.ann_data = ann_data
    return data_module



def convert_labels(labels):
    """
    convert labels to 0,1, 2, ...
    :param labels:
    :return:
    """
    label_dict = dict()
    for i, label in enumerate(np.unique(labels)):
        label_dict[label] = i
    new_labels = np.zeros_like(labels)
    for i, label in enumerate(labels):
        new_labels[i] = label_dict[label]
    return new_labels


class STAMarker:
    def __init__(self, n, save_dir, config, logging_level=logging.INFO):
        """
        n: int, number of graph attention auto-econders to train
        save_dir: directory to save the models
        config: config file for training
        """
        self.n = n
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            logging.info("Create save directory {}".format(save_dir))
        self.version_dirs = [os.path.join(save_dir, f"version_{i}") for i in range(n)]
        self.config = config
        self.logger = logging.getLogger("STAMarker")
        self.logger.setLevel(logging_level)
        self.consensus_labels = None

    def train_auto_encoders(self, data_module):
        for seed in range(self.n):
            self._train_auto_encoder(data_module, seed, self.config)
        self.logger.info("Finished training {} auto-encoders".format(self.n))

    def clustering(self, data_module, cluster_method, cluster_params):
        """
        Cluster the latent space of the trained auto-encoders
        Cluster method should be "louvain" or "mclust"
        """
        for version_dir in self.version_dirs:
            self._clustering(data_module, version_dir, cluster_method, cluster_params)
        self.logger.info("Finished {} clustering with {}".format(self.n, cluster_method))

    def _clustering(self, data_module, version_dir, cluster_method, cluster_params):
        """
        Cluster the latent space of the trained auto-encoder
        """
        if cluster_method == "louvain":
            run_louvain(data_module, version_dir, cluster_params)
        elif cluster_method == "mclust":
            run_mclust(data_module, version_dir, cluster_params)
        else:
            raise ValueError("Unknown clustering method")

    def consensus_clustering(self, n_clusters, name="cluster_labels.npy", show_plot=False):
        label_files = glob.glob(self.save_dir + f"/version_*/{name}")
        labels_list = list(map(lambda file: np.load(file), label_files))
        labels_list = np.vstack(labels_list)
        cons_mat = compute_consensus_matrix(labels_list)
        if show_plot:
            figure, consensus_labels = plot_clustered_consensus_matrix(cons_mat, n_clusters)
            figure.savefig(os.path.join(self.save_dir,
                                        "consensus_clustering_{}_clusters.png".format(n_clusters)), dpi=300)
            print("Save consensus clustering plot to {}".format(os.path.join(self.save_dir, "consensus_clustering.png")))
            # delete the figure
            del figure
        else:
            linkage_matrix = hierarchy.linkage(cons_mat, method='average', metric='euclidean')
            consensus_labels = hierarchy.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
        consensus_labels = convert_labels(consensus_labels)
        self.consensus_labels = consensus_labels    
        np.save(os.path.join(self.save_dir, "consensus_labels.npy"), consensus_labels)
        return consensus_labels

    def train_classifiers(self, data_module, n_class,
                          consensus_labels_path="consensus_labels.npy"):
        target_y = np.load(os.path.join(self.save_dir, consensus_labels_path))
        for seed, version_dir in enumerate(self.version_dirs):
            self._train_classifier(data_module, version_dir, target_y, self.config, n_class, seed=seed)
        self.logger.info("Finished training {} classifiers".format(self.n))

    def _train_classifier(self, data_module, version_dir, target_y, config, n_classes, seed=None):
        timer = Timer()
        pl.seed_everything(seed)
        rep_dim = config["stagate"]["params"]["hidden_dims"][-1]
        stagate = intSTAGATE.load_from_checkpoint(os.path.join(version_dir, "checkpoints", "stagate.ckpt"))
        classifier = StackClassifier(rep_dim, n_classes=n_classes, architecture="MLP")
        classifier.prepare(stagate, data_module.train_dataset, target_y,
                           balanced=config["mlp"]["balanced"], test_prop=config["mlp"]["test_prop"])
        classifier.set_optimizer_params(config["mlp"]["optimizer"], config["mlp"]["scheduler"])
        logger = TensorBoardLogger(save_dir=self.save_dir, name=None,
                                   default_hp_metric=False,
                                   version=seed)
        trainer = pl.Trainer(logger=logger, **config["classifier_trainer"])
        timer.tic("clf")
        trainer.fit(classifier)
        clf_time = timer.toc("clf")
        with open(os.path.join(version_dir, "runtime.csv"), "a+") as f:
            f.write("\n")
            f.write("{}, clf_time, {:.2f}, ".format(seed, clf_time / 60))
        trainer.save_checkpoint(os.path.join(version_dir, "checkpoints", "mlp.ckpt"))
        target_y = classifier.dataset.target_y.numpy()
        # all_props = class_proportions(target_y)
        # val_props = class_proportions(target_y[classifier.val_dataset.indices])
        # if self.logger.level == logging.DEBUG:
        #     print("All class proportions " + "|".join(["{:.2f}%".format(prop * 100) for prop in all_props]))
        #     print("Val class proportions " + "|".join(["{:.2f}%".format(prop * 100) for prop in val_props]))
        # np.save(os.path.join(version_dir, "confusion.npy"), classifier.confusion)

    def compute_smaps(self, data_module, return_recon=False, normalize=False):
        smaps = []
        if return_recon:
            recons = []
        for version_dir in self.version_dirs:
            if return_recon:
                smap, recon = self._compute_smap(data_module, version_dir, return_recon=return_recon)
                smaps.append(smap)
                recons.append(recon)
            else:
                smap = self._compute_smap(data_module, version_dir, return_recon=return_recon)
                smaps.append(smap)
        smaps = np.array(smaps).mean(axis=0)
        smaps = pd.DataFrame(smaps, columns=data_module.ann_data.var.index)
        if return_recon:
            recons = np.array(recons).mean(axis=0)
            return smaps, recons
        else:
            return smaps
        self.logger.info("Finished computing {} smaps".format(self.n))

    def _compute_smap_zscore(self, smap, labels, logtransform=False):
        scores = np.log(smap + 1) if logtransform else copy.copy(smap)
        unique_labels = np.unique(labels)
        for l in unique_labels:
            scores[labels == l, :] = scipy.stats.zscore(scores[labels == l, :], axis=1)
        return scores

    def _train_auto_encoder(self, data_module, seed, config):
        """
        Train a single graph attention auto-encoder
        """
        pl.seed_everything(seed)
        version = f"version_{seed}"
        version_dir = os.path.join(self.save_dir, version)
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
        os.makedirs(version_dir, exist_ok=True)
        logger = TensorBoardLogger(save_dir=self.save_dir, name=None,
                                   default_hp_metric=False,
                                   version=seed)
        model = intSTAGATE(**config["stagate"]["params"])
        model.set_optimizer_params(config["stagate"]["optimizer"], config["stagate"]["scheduler"])
        trainer = pl.Trainer(logger=logger, **config["stagate_trainer"])
        timer = Timer()
        timer.tic("fit")
        trainer.fit(model, data_module)
        fit_time = timer.toc("fit")
        with open(os.path.join(version_dir, "runtime.csv"), "w+") as f:
            f.write("{}, fit_time, {:.2f}, ".format(seed, fit_time / 60))
        trainer.save_checkpoint(os.path.join(version_dir, "checkpoints", "stagate.ckpt"))
        del model, trainer
        if config["stagate_trainer"]["gpus"] > 0:
            torch.cuda.empty_cache()
        logging.info(f"Finshed running version {seed}")

    def _compute_smap(self, data_module, version_dir, return_recon=True):
        """
        Compute the saliency map of the trained auto-encoder
        """
        stagate = intSTAGATE.load_from_checkpoint(os.path.join(version_dir, "checkpoints", "stagate.ckpt"))
        cls = StackClassifier.load_from_checkpoint(os.path.join(version_dir, "checkpoints", "mlp.ckpt"))
        stagate_cls = STAGATEClsModule(stagate.model, cls.model)
        smap, _ = stagate_cls.get_saliency_map(data_module.train_dataset.x,
                                               data_module.train_dataset.edge_index)
        smap = F.relu(smap).cpu().detach().numpy()  # filter out zeros
        if return_recon:
            recon = stagate(data_module.train_dataset.x, data_module.train_dataset.edge_index)[1].cpu().detach().numpy()
            return smap, recon
        else:
            return smap


def run_louvain(data_module, version_dir, resolution, name="cluster_labels"):
    """
    Run louvain clustering on the data_module
    """
    stagate = intSTAGATE.load_from_checkpoint(os.path.join(version_dir, "checkpoints", "stagate.ckpt"))
    embedding = stagate(data_module.train_dataset.x, data_module.train_dataset.edge_index)[0].cpu().detach().numpy()
    ann_data = copy.copy(data_module.ann_data)
    ann_data.obsm["stagate"] = embedding
    sc.pp.neighbors(ann_data, use_rep='stagate')
    sc.tl.louvain(ann_data, resolution=resolution)
    save_path = os.path.join(version_dir, "{}.npy".format(name))
    np.save(save_path, ann_data.obs["louvain"].to_numpy().astype("int"))
    print("Save louvain results to {}".format(save_path))


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


def run_mclust(data_module, version_dir, n_clusters, name="cluster_labels"):
    stagate = intSTAGATE.load_from_checkpoint(os.path.join(version_dir, "checkpoints", "stagate.ckpt"))
    embedding = stagate(data_module.train_dataset.x, data_module.train_dataset.edge_index)[0].cpu().detach().numpy()
    labels = mclust_R(embedding, n_clusters)
    save_path = os.path.join(version_dir, "{}.npy".format(name))
    np.save(save_path, labels.astype("int"))
    print("Save MClust results to {}".format(save_path))


def class_proportions(target):
    n_classes = len(np.unique(target))
    props = np.array([np.sum(target == i) for i in range(n_classes)])
    return props / np.sum(props)
