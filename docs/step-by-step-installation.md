# Step-by-step guide to create conda environment
-  Create a new conda environment with the name `myenv` and python version 3.7
``` bash
conda create -n marker python=3.8 ipython jupyter
```
-  Activate the environment
``` bash
conda activate myenv
```

- Install Pytorch
``` bash
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia
```

- Install PyTorch-Lightning
``` bash
pip install pytorch-lightning==1.5.8
```

- Install Scanpy and dependencies
``` bash
conda install -c conda-forge scanpy python-igraph leidenalg
```

- Install torch geometric
``` bash
pip install torch_geometric
pip install --no-index torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
```

- Install scikit-misc
``` bash
pip install --user scikit-misc
```
- Install rpy2 for mclust clustering

``` bash
conda install -c conda-forge rpy2
```
Now, you have successfully set up the Conda environment for STAMarker.
Make sure to activate the environment whenever you work on your project using the following command:
```
conda activate marker
```
