from abc import ABC
from typing import Any, List
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch_geometric.data import Data
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from .modules import STAGATEModule, StackMLPModule
from .dataset import RepDataset, Batch
from utils import Timer

def get_optimizer(name):
    if name == "ADAM":
        return torch.optim.Adam
    elif name == "ADAGRAD":
        return torch.optim.Adagrad
    elif name == "ADADELTA":
        return torch.optim.Adadelta
    elif name == "RMS":
        return torch.optim.RMSprop
    elif name == "ASGD":
        return torch.optim.ASGD
    else:
        raise NotImplementedError


def get_scheduler(name):
    if name == "STEP_LR":
        return torch.optim.lr_scheduler.StepLR
    elif name == "EXP_LR":
        return torch.optim.lr_scheduler.ExponentialLR
    else:
        raise NotImplementedError

class BaseModule(pl.LightningModule, ABC):
    def __init__(self):
        super(BaseModule, self).__init__()
        self.optimizer_params = None
        self.scheduler_params = None
        self.model = None
        self.timer = Timer()
        self.automatic_optimization = False

    def set_optimizer_params(self,
                             optimizer_params: dict,
                             scheduler_params: dict):
        self.optimizer_params = optimizer_params
        self.scheduler_params = scheduler_params

    def configure_optimizers(self):
        optimizer = get_optimizer(self.optimizer_params["name"])(
            self.model.parameters(),
            **self.optimizer_params["params"])
        scheduler = get_scheduler(self.scheduler_params["name"])(optimizer, **self.scheduler_params["params"])
        return [optimizer], [scheduler]

    def on_train_epoch_start(self) -> None:
        self.timer.tic('train')


class intSTAGATE(BaseModule):
    """
    intSTAGATE Lightning Module
    """
    def __init__(self,
                 in_features: int = None,
                 hidden_dims: List[int] = None,
                 gradient_clipping: float = 5.0,
                 **kwargs):
        super(intSTAGATE, self).__init__()
        self.model = STAGATEModule(in_features, hidden_dims)
        self.auto_encoder_epochs = None
        self.gradient_clipping = gradient_clipping
        self.pred_labels = None
        self.save_hyperparameters()

    def configure_optimizers(self) -> (dict, dict):
        auto_encoder_optimizer = get_optimizer(self.optimizer_params["name"])(
            list(self.model.parameters()),
            **self.optimizer_params["params"])
        auto_encoder_scheduler = get_scheduler(self.scheduler_params["name"])(auto_encoder_optimizer,
                                                                              **self.scheduler_params["params"])
        return [auto_encoder_optimizer], [auto_encoder_scheduler]

    def forward(self, x, edge_index) -> Any:
        return self.model(x, edge_index)

    def training_step(self, batch, batch_idx):
        batch = batch.to(self.device)
        opt_auto_encoder = self.optimizers()
        z, x_hat = self.model(batch.x, batch.edge_index)
        loss = F.mse_loss(batch.x, x_hat)
        opt_auto_encoder.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clipping)
        opt_auto_encoder.step()
        self.log("Training auto-encoder|Reconstruction errors", loss.item(), prog_bar=True)
        self.logger.experiment.add_scalar('auto_encoder/loss', loss.item(), self.current_epoch)

    def on_train_epoch_end(self) -> None:
        time = self.timer.toc('train')
        sch_auto_encoder = self.lr_schedulers()
        sch_auto_encoder.step()
        self.logger.experiment.add_scalar('train_time', time, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        pass


def _compute_correct(scores, target_y):
    _, pred_labels = torch.max(scores, axis=1)
    correct = (pred_labels == target_y).sum().item()
    return pred_labels, correct


class CoordTransformer(object):
    def __init__(self, coord):
        self.coord = coord

    def transform(self):
        factor = np.max(np.max(self.coord, axis=0) - np.min(self.coord, axis=0))
        return (self.coord - np.min(self.coord, axis=0)) / factor


class StackClassifier(BaseModule):
    def __init__(self, in_features: int,
                 n_classes: int = 7,
                 batch_size: int = 1000,
                 shuffle: bool = False,
                 hidden_dims: List[int] = [30],
                 architecture: str = "MLP",
                 sta_path: str = None,
                 **kwargs):
        super(StackClassifier, self).__init__()
        self.in_features = in_features
        self.architecture = architecture
        self.batch_size = batch_size
        self.shuffle = shuffle
        if architecture == "MLP":
            self.model = StackMLPModule(in_features, n_classes, hidden_dims, **kwargs)
        else:
            raise NotImplementedError
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.automatic_optimization = False
        self.sampler = None
        self.test_prop = None
        self.confusion = None
        self.balanced = None
        self.save_hyperparameters()

    def prepare(self,
                stagate: intSTAGATE,
                dataset: Data,
                target_y,
                test_prop: float = 0.5,
                balanced: bool = True):
        self.balanced = balanced
        self.test_prop = test_prop
        with torch.no_grad():
            representation, _ = stagate(dataset.x, dataset.edge_index)
        if hasattr(dataset, "ground_truth"):
            ground_truth = dataset.ground_truth
        else:
            ground_truth = None
        if isinstance(target_y, np.ndarray):
            target_y = torch.from_numpy(target_y).type(torch.LongTensor)
        elif isinstance(target_y, torch.Tensor):
            target_y = target_y.type(torch.LongTensor)
        else:
            raise TypeError("target_y must be either a torch tensor or a numpy ndarray.")
        self.dataset = RepDataset(representation, target_y, ground_truth=ground_truth)
        n_val = int(len(self.dataset) * test_prop)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(
            self.dataset, [len(self.dataset) - n_val, n_val])
        if balanced:
            target_y = target_y[self.train_dataset.indices]
            class_sample_count = np.array([len(np.where(target_y == t)[0]) for t in np.unique(target_y)])
            weight = 1. / class_sample_count
            samples_weight = np.array([weight[t] for t in target_y])
            samples_weight = torch.from_numpy(samples_weight)
            samples_weight = samples_weight.double()
            self.sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    def forward(self, x, edge_index=None) -> Any:
        if self.architecture == "MLP":
            return self.model(x)
        elif self.architecture == "STACls":
            _, output = self.model(x, edge_index)
            return output

    def training_step(self, batch, batch_idx):
        batch = Batch(**batch)
        batch = batch.to(self.device)
        opt = self.optimizers()
        opt.zero_grad()
        output = self.model(batch.x)
        loss = F.cross_entropy(output["score"], batch.y)
        self.manual_backward(loss)
        opt.step()
        _, correct = _compute_correct(output["score"], batch.y)
        self.log(f"Training {self.architecture} classifier|Cross entropy", loss.item(), prog_bar=True)
        return {"loss": loss, "correct": correct}

    def training_epoch_end(self, outputs):
        time = self.timer.toc('train')
        self.logger.experiment.add_scalar(f'classifier-{self.architecture}/train_time', time, self.current_epoch)
        all_loss = torch.stack([x["loss"] for x in outputs])
        all_correct = np.sum([x["correct"] for x in outputs])
        train_acc = all_correct / len(self.train_dataset)
        self.logger.experiment.add_scalar(f'classifier-{self.architecture}/loss',
                                          torch.mean(all_loss), self.current_epoch)
        self.logger.experiment.add_scalar(f'classifier-{self.architecture}/train_acc',
                                          train_acc, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        batch = Batch(**batch)
        batch = batch.to(self.device)
        with torch.no_grad():
            output = self.model(batch.x)
            loss = F.cross_entropy(output["score"], batch.y)
        pred_labels, correct = _compute_correct(output["score"], batch.y)
        return {"loss": loss, "correct": correct, "pred_labels": pred_labels, "true_labels": batch.y}

    def validation_epoch_end(self, outputs):
        all_loss = torch.stack([x["loss"] for x in outputs])
        all_correct = np.sum([x["correct"] for x in outputs])
        pred_labels = torch.cat([x["pred_labels"] for x in outputs]).cpu().detach().numpy()
        true_labels = torch.cat([x["true_labels"] for x in outputs]).cpu().detach().numpy()
        confusion = confusion_matrix(true_labels, pred_labels)
        val_acc = all_correct / len(self.val_dataset)
        self.logger.experiment.add_scalar(f'classifier-{self.architecture}/val_loss',
                                          torch.mean(all_loss), self.current_epoch)
        self.logger.experiment.add_scalar(f'classifier-{self.architecture}/val_acc',
                                          val_acc, self.current_epoch)
        print("\n validation ACC={:.4f}".format(val_acc))
        print(confusion)
        self.confusion = confusion

    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=self.batch_size, sampler=self.sampler)
        return loader

    def val_dataloader(self):
        loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=self.shuffle, drop_last=False)
        return loader

    def test_dataloader(self):
        raise NotImplementedError

    def predict_dataloader(self):
        raise NotImplementedError