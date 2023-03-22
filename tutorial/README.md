# Model configuration  
The folder `_params` contains two yaml files `model.yaml` and `trainer.yaml` that specify the model setttings and training procedure respectively. \
Please update the configuration accordingly.

## Model specification
Configuaration of the autoencoder
``` yaml
stagate:
  name: intSTAGATE
  params:
      in_features: 3000
      architecture: MLP
      hidden_dims: [512, 30]
      gradient_clipping: 5.0
      
  optimizer:
    name: ADAM
    params:
      lr: 0.0001
      weight_decay: 0.0001
  scheduler:
    name: STEP_LR
    params:
      step_size: 1000
      gamma: 0.9
```
- `in_feature`: the dimension of the input features
- `hidden_dims`: a list indicates the number of neurons of the hidden layers
- `gradient_clipping`: float that controls the optimization
- `optimizer`: configuration of the optimizers

Configuaration of the clustering method
``` yaml 
clustering:
  r_seed: 2020
  n_clusters: 7
  model_name: EEE
```

Configuaration of the classifiers
``` yaml
mlp:
  balanced: True
  test_prop: 0.1
  target_class: 3
  optimizer:
    name: ADAM
    params:
      lr: 0.001
      weight_decay: 0.0001
  scheduler:
    name: STEP_LR
    params:
      step_size: 250
      gamma: 0.5
 ```
 ## Trainer specification
Trainer for the autoencoder.

``` yaml
stagate_trainer:
  max_epochs: 1000  # Maximum number of epochs to train
  min_epochs: 500  # Min number of epochs to train
  limit_train_batches: 1.0
  limit_val_batches: 1.0
  limit_test_batches: 1.0
  check_val_every_n_epoch: 100
  log_every_n_steps: 2 # How often to add logging rows (does not write to disk)
  precision: 32
  detect_anomaly: True
  auto_select_gpus: True
  enable_model_summary: False
  gpus: 1 # number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
  num_sanity_val_steps: 10
  track_grad_norm: -1 # Otherwise tracks that norm (2 for 2-norm)
  enable_checkpointing: False
 ```
- `max_epochs` and `min_epochs` controls the number of the trained epochs. We recommend set > 500. We used `max_epochs: 1000` in all the experiments.
- Please refer to the documentation of `pytorch.lightning` for a detailed decription.


Trainer for the classifier.
```yaml
classifier_trainer:
  max_epochs: 300 # Maximum number of epochs to train
  min_epochs: 300 # Min number of epochs to train
  limit_train_batches: 1.0
  limit_val_batches: 1.0
  limit_test_batches: 1.0
  check_val_every_n_epoch: 100
  log_every_n_steps: 2 # How often to add logging rows (does not write to disk)
  precision: 32
  detect_anomaly: True
  gpus: 1 # number of gpus to train on (int) or which GPUs to train on (list or str) applied per node
  auto_select_gpus: True
  enable_model_summary: False
  num_sanity_val_steps: 10
  track_grad_norm: -1 # Otherwise tracks that norm (2 for 2-norm)
  enable_checkpointing: False
```
