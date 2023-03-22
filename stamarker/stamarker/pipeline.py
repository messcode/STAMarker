import pytorch_lightning as pl
import copy
import torch
import os
import shutil
import logging
import glob
import sys
import numpy as np
import scipy
import scanpy as sc
from pytorch_lightning.loggers import TensorBoardLogger
from scipy.cluster import hierarchy
from .models import intSTAGATE, StackClassifier
from .utils import plot_consensus_map, consensus_matrix, Timer
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

    def load_from_dir(self, save_dir, ):
        """
        Load the trained models from a directory
        """
        self.version_dirs = glob.glob(os.path.join(save_dir, "version_*"))
        self.version_dirs = sorted(self.version_dirs, key=lambda x: int(x.split("_")[-1]))
        # check if all version dir have `checkpoints/stagate.ckpt`
        version_dirs_valid = []
        for version_dir in self.version_dirs:
            if not os.path.exists(os.path.join(version_dir, "checkpoints/stagate.ckpt")):
                self.logger.warning("No checkpoint found in {}".format(version_dir))
            else:
                version_dirs_valid.append(version_dir)
        self.version_dirs = version_dirs_valid
        self.logger.info("Load {} autoencoder models from {}".format(len(version_dirs_valid), save_dir))
        # check if all version dir have `cluster_labels.npy` raise warning if not
        missing_cluster_labels = False
        for version_dir in self.version_dirs:
            if not os.path.exists(os.path.join(version_dir, "cluster_labels.npy")):
                missing_cluster_labels = True
                msg = "No cluster labels found in {}.".format(version_dir)
                self.logger.warning(msg)
        if missing_cluster_labels:
            self.logger.warning("Please run clustering first.")
        # check if save_dir has `consensus.npy` raise warning if not
        if not os.path.exists(os.path.join(save_dir, "consensus.npy")):
            self.logger.warning("No consensus labels found in {}".format(save_dir))
        else:
            self.consensus_labels = np.load(os.path.join(save_dir, "consensus.npy"))
        # check if all version dir have `checkpoints/mlp.ckpt` raise warning if not
        missing_clf = False
        for version_dir in self.version_dirs:
            if not os.path.exists(os.path.join(version_dir, "checkpoints/mlp.ckpt")):
                self.logger.warning("No classifier checkpoint found in {}".format(version_dir))
                missing_clf = True
        if missing_clf:
            self.logger.warning("Please run classifier training first.")
        if not missing_cluster_labels and not missing_clf:
            self.logger.info("All models are trained and ready to use.")

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

    def consensus_clustering(self, n_clusters, name="cluster_labels.npy"):
        sys.setrecursionlimit(100000)
        label_files = glob.glob(self.save_dir + f"/version_*/{name}")
        labels_list = list(map(lambda file: np.load(file), label_files))
        cons_mat = consensus_matrix(labels_list)
        row_linkage, _, figure = plot_consensus_map(cons_mat, return_linkage=True)
        figure.savefig(os.path.join(self.save_dir, "consensus_clustering.png"), dpi=300)
        consensus_labels = hierarchy.cut_tree(row_linkage, n_clusters).squeeze()
        np.save(os.path.join(self.save_dir, "consensus"), consensus_labels)
        self.consensus_labels = consensus_labels
        self.logger.info("Save consensus labels to {}".format(os.path.join(self.save_dir, "consensus.npz")))

    def train_classifiers(self, data_module, n_clusters, name="cluster_labels.npy"):
        for i, version_dir in enumerate(self.version_dirs):
            # _train_classifier(self, data_module, version_dir, target_y, n_classes, seed=None)
            self._train_classifier(data_module, version_dir, self.consensus_labels, 
                                   n_clusters, self.config, seed=i)
        self.logger.info("Finished training {} classifiers".format(self.n))

    def compute_smaps(self, data_module, return_recon=True, normalize=True):
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
        if return_recon:
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

    def _train_classifier(self, data_module, version_dir, target_y, n_classes, config, seed=None):
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
        all_props = class_proportions(target_y)
        val_props = class_proportions(target_y[classifier.val_dataset.indices])
        if self.logger.level == logging.DEBUG:
            print("All class proportions " + "|".join(["{:.2f}%".format(prop * 100) for prop in all_props]))
            print("Val class proportions " + "|".join(["{:.2f}%".format(prop * 100) for prop in val_props]))
        np.save(os.path.join(version_dir, "confusion.npy"), classifier.confusion)

    def _compute_smap(self, data_module, version_dir, return_recon=True):
        """
        Compute the saliency map of the trained auto-encoder
        """
        stagate = intSTAGATE.load_from_checkpoint(os.path.join(version_dir, "checkpoints", "stagate.ckpt"))
        cls = StackClassifier.load_from_checkpoint(os.path.join(version_dir, "checkpoints", "mlp.ckpt"))
        stagate_cls = STAGATEClsModule(stagate.model, cls.model)
        smap, _ = stagate_cls.get_saliency_map(data_module.train_dataset.x,
                                               data_module.train_dataset.edge_index)
        smap = smap.detach().cpu().numpy()
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







