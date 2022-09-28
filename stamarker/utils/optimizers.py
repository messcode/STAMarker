import torch


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
