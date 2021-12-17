
# Hand Pose Estimation in PyTorch

This repository hosts the codes for hand pose estimation of the team from University of West Bohemia,
New Technologies for Information Society - NTIS, Pilsen, Czech Republic.

The implementation follows the research https://arxiv.org/abs/1711.07399 of Moon et al.

**V2V-PoseNet: Voxel-to-Voxel Prediction Network for Accurate 3D Hand and Human Pose Estimation from a Single Depth Map**

## Pre-requisites

- Python 3.8 (tested, lower versions may be possible)
- PyTorch
- Windows or Linux (tested)

All the dependencies are listed in *requirements.txt*, however not all are required to train a model.
Some dependencies are used to visualize the results (script *HandPoseEstimation/debugging.py*). Many
are installed with the requirement *mayavi*. 

## Instalation

Clone this repo, install requirements, and you are good to go!

`pip install -r requirements.txt`

There is no need to install anything other than the required dependencies.
However, the *pytorch* framework is listed with a dependency on CUDA 10.1.
If you have other version of CUDA on your system, you must modify the installation process of PyTorch.
The scripts support CPU inference, however GPU is required for training.
Single- and Multi-node training is possible. Multi-node computation is supported via the NCCL backend
and thus requires GPUs with CUDA support. 

### Windows

Windows users will have to install PyTorch manually from a wheel, eg:

`pip3 install torch==1.7.0+cu101 torchvision==0.8.1+cu101 -f https://download.pytorch.org/whl/cu101/torch_stable.html`

## Training

The input data of the V2V Pose Net is a 3D cube of voxels that is regressed into per-joint estimation of location
heatmaps. Training is possible, however some expectations must be fulfilled. The data are expected in a certain format.

The input data must be in a HDF5 dataset file (h5py) with keys:

`data[key][index]=numpy.array(dtype=bytes)`

The `key` is defined in the arguments of training/testing script (e.g. "real_voxels"). The value of the key 
is an array of bytes that is a 3D numpy array with ones and zeros, representing the voxels of a surface of a
hand. **TODO: data scripts**. The `index` is a string (e.g. "0", "1", ...). This is because the values are an
array of bytes with non-consistent lengths, which cannot be represented as N-D array.

The labels are represented as relative 3D coordinates of joint locations inside the input 3D cube, 
with limits of [-1; 1].

`data["labels"][index]=numpy.array((n_joints, 3), dtype=float)`

The key is always "labels", the index is an integer.

Furthermore, we provide an option to set a different size of the input cubes for each handshape.

`data[cubes_key][index]=numpy.array((3,), dtype=float)`

Again, the `cubes_key` is set in the arguments. The index is an integer.

**TODO: Sample training data**

### Run Training

The training is supported in several variants. **(1) Single node, single GPU, (2) Single node, multi GPU, (3)
Multi node, single GPU**. All the variants are run as a module.

**Single node, single GPU**

`python -m hpoes_pytorch.HandPoseEstimation.train [TRAIN_H5] [OUTPUT_MODEL_H5]`

**Single node, multi GPU**

`python -m hpoes_pytorch.HandPoseEstimation.train [TRAIN_H5] [OUTPUT_MODEL_H5] --multi_node_params
rank world_size`

This command needs to be run for each GPU on the node that we want to use. Ranks are from 0 to N-1, where N is the number of
GPUs we want to use. The `world_size` is the number of processes N.

**Multi node, single GPU**

`mpirun -n "$nr_gpus"`

## The Team