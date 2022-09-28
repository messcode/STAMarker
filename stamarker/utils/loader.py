import numpy as np
import os
from models.stagate import  intSTAGATE


def load_model(version_dir, name="stagate"):
    if name == "stagate":
        model = intSTAGATE.load_from_checkpoint(os.path.join(version_dir, "checkpoints", "stagate.ckpt"))
    else:
        raise ValueError("Unknown model name")
    return model


def load_metric(version_dir, name="confusion", test_prop=0.1):
    if name == "mul_confusion":
        return np.load(os.path.join(version_dir, f"classifier-confusion-balanced-{test_prop:.2f}.npz.npy"))
    elif name == "bin_confusion":
        return np.load(os.path.join(version_dir, f"binary_classifier-confusion-balanced-{test_prop:.2f}.npz.npy"))
    elif name == "clustering_labels":
        return np.load(os.path.join(version_dir, "final_mclust.npy"))
    else:
        raise ValueError("Unknown model name")