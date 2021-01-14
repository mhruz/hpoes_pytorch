from .architectures import V2VModel as V2V
from Utils import v2v_misc
import h5py
import numpy as np
import argparse
import time
import io
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist


def train_net_on_node(local_rank, global_rank_offset, world_size, args):
    rank = local_rank + global_rank_offset
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    # create multiple models on multiple GPUs
    model = DDP(V2V(1, numJoints), device_ids=[local_rank])
    # Choose an optimizer algorithm
    optimizer = optim.RMSprop(model.parameters(), lr=0.00025)
    # choose criterion
    criterion = nn.MSELoss()

    if args.init_net is not None:
        # make sure the model was saved by process with rank 0
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}

        checkpoint = torch.load(args.init_net, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_idx = checkpoint["epoch"]
        loss = checkpoint["loss"]


if __name__ == '__main__':
    # parse commandline
    parser = argparse.ArgumentParser(description='Train the V2V-PoseNet on Volumetric data.')
    parser.add_argument('train_h5', type=str, help='path to training h5 file with non-augmented data')
    parser.add_argument('--init_net', type=str,
                        help='path to pre-trained network (h5), if you want to continue training')
    parser.add_argument('--max_epoch', type=int, help='number of epochs to perform, default value = 10', default=10)
    parser.add_argument('--batch_size', type=int, help='batch size in each iteration, default value = 8', default=8)
    parser.add_argument('--dev_h5', type=str, help='path to development h5 file with non-augmented data')
    parser.add_argument('--log', type=str, help='path to output log file')
    parser.add_argument('--save_iter', type=int, help='interval of saving a model (in epochs), default = 1', default=1)
    parser.add_argument('--data_label', type=str, help='the label of data in the H5 file, default = real_voxels',
                        default='real_voxels')
    parser.add_argument('--read_data_to_memory', type=bool,
                        help='whether to read all the training data to memory, only '
                             'use for reasonable small data (< RAM)')
    parser.add_argument('output', type=str, help='name of the output model')
    args = parser.parse_args()

    batch_size = args.batch_size
    max_epochs = args.max_epoch
    epoch_idx = 0

    augment = True
    logging = False
    dev = False

    if args.log is not None:
        f_log = open(args.log, 'wt')
        logging = True

    f_train = h5py.File(args.train_h5)
    # determine the number of joints
    numJoints = f_train['labels'].shape[1]

    # when training on multi-node environment, make sure these environment variables were set before running script
    gpus_per_node = int(os.environ['PBS_NGPUS'])
    world_size = int(os.environ['OMPI_COMM_WORLD_SIZE']) * gpus_per_node
    node_id = int(os.environ['OMPI_COMM_WORLD_RANK'])

    # the limits of global ranks of the processes to be run on this node
    ranks = (node_id * gpus_per_node, (node_id + 1) * gpus_per_node)
    global_rank_offset = ranks[0]

    mp.spawn(train_net_on_node, args=(global_rank_offset, world_size, args))

    key = args.data_label

    if args.dev_h5 is not None:
        f_dev = h5py.File(args.dev_h5)
        dev = True

    # get the data
    if args.read_data_to_memory is not None:
        if logging:
            f_log.write('Reading training data...\n')
            s = time.time()

        num_samples = len(f_train[key])
        data_train = {}
        data_train['cubes'] = f_train['cubes'][:]
        data_train['labels'] = f_train['labels'][:]
        data_train[key] = {}
        for i in range(num_samples):
            data_train[key][str(i)] = f_train[key][str(i)][:]

        if logging:
            e = time.time()
            f_log.write('Training data read in {} s\n'.format(e - s))

        if args.dev_h5 is not None:
            if logging:
                f_log.write('Reading dev data...\n')
                s = time.time()

            num_samples = len(f_train[key])
            data_dev = {}
            data_dev['cubes'] = f_dev['cubes'][:]
            data_dev['labels'] = f_dev['labels'][:]
            data_dev[key] = {}
            for i in range(num_samples):
                data_dev[key][str(i)] = f_dev[key][str(i)][:]

            if logging:
                e = time.time()
                f_log.write('Dev data read in {} s\n'.format(e - s))
    else:
        data_train = f_train
        if args.dev_h5 is not None:
            data_dev = f_dev

    # get the indexes of data for later shuffling
    indexes_all = list(range(len(f_train[key].keys())))

    while epoch_idx < max_epochs:
        if logging:
            f_log.write('Epoch {}\n'.format(epoch_idx))
            # compute the loss on dev data if available
            last_idx = 0

        if dev:
            dev_loss = []
            num_samples = len(data_dev[key])
            model.eval()

            for i in range(0, num_samples, batch_size):
                batch_data = []
                batch_labels = []
                for idx in range(min(batch_size, num_samples - i)):
                    pdata = data_dev[key][str(i)][:].tostring()
                    _file = io.BytesIO(pdata)
                    data = np.load(_file)['arr_0']

                    batch_data.append(data)
                    batch_labels.append(data_dev['labels'][idx + i])

                batch_data = np.array(batch_data, dtype=np.float32)
                batch_labels = np.array(batch_labels)
                batch_data = np.expand_dims(batch_data, 1)

                model

                # divide the data
                batch_data_gpu = []
                target_gpu = []

                one_gpu_load = batch_data.shape[0] // args.number_of_gpus
                for g in range(args.number_of_gpus):
                    batch_data_gpu.append(batch_data[g * one_gpu_load:(g + 1) * one_gpu_load])
                    target_gpu.append(v2v_misc.makeHeatMapsGPU(batch_labels, device=g))

                # forward pass through the net
                for n, m in enumerate(models):
                    pred = m(to_gpu(batch_data_gpu[n]))
                    # calculate loss
                    loss = F.mean_squared_error(pred, target_gpu[n])
                    dev_loss.append(to_cpu(loss.data))

            dev_loss = np.mean(np.array(dev_loss))
            f_log.write('Dev loss: {}\n'.format(dev_loss))
            f_log.flush()

        # in every epoch shuffle the indexes of data
        np.random.shuffle(indexes_all)
        num_samples = len(data_train[key])
        with chainer.using_config('train', True):
            for i in range(0, num_samples, batch_size):
                batch_data = []
                batch_labels = []
                cubes = []
                if min(batch_size, num_samples - i) < 2:
                    continue

                for idx in range(min(batch_size, num_samples - i)):
                    sdata = data_train[key][str(indexes_all[idx + i])][:].tostring()
                    _file = io.BytesIO(sdata)
                    data = binvox_rw.read_as_3d_array(_file).data
                    data = data.astype(np.float32)

                    batch_data.append(data)
                    batch_labels.append(data_train['labels'][indexes_all[idx + i]])

                    cubes.append(data_train['cubes'][indexes_all[idx + i]])

                (batch_data, batch_labels) = v2v_misc.augmentation_volumetric(batch_data, batch_labels, cubes)
                batch_data = np.expand_dims(batch_data, 1)

                # divide the data
                batch_data_gpu = []
                target_gpu = []

                one_gpu_load = batch_data.shape[0] // args.number_of_gpus
                for g in range(args.number_of_gpus):
                    batch_data_gpu.append(batch_data[g * one_gpu_load:(g + 1) * one_gpu_load])
                    target_gpu.append(
                        v2v_misc.makeHeatMapsGPU(batch_labels[g * one_gpu_load:(g + 1) * one_gpu_load], device=g))

                # forward pass through the nets
                loss = []

                for n, m in enumerate(models):
                    print("device={}".format(n))
                    device_data = chainer.Variable(to_gpu(batch_data_gpu[n], device=n))
                    print(device_data.data.device)
                    print(device_data.data.shape)

                    # calculate loss
                    loss.append(F.mean_squared_error(m(device_data), target_gpu[n]))

                if logging:
                    for n, l in enumerate(loss):
                        f_log.write('loss on GPU {}: {}\n'.format(n, to_cpu(l.data)))

                    f_log.flush()

                # Calculate the gradients in the network
                for m in models:
                    m.cleargrads()

                for l in loss:
                    l.backward()

                for n in range(1, args.number_of_gpus):
                    models[0].addgrads(models[n])

                # Update all the trainable paremters
                optimizer.update()

                for n in range(1, args.number_of_gpus):
                    models[n].copyparams(models[0])

        epoch_idx += 1

        # save the intermediate net
        if epoch_idx % args.save_iter == 0:
            filename = args.output + '_epoch_' + str(epoch_idx) + '.h5'
            serializers.save_hdf5(filename, models[0])
            os.chmod(filename, 0o777)

    if logging:
        f_log.close()

    # save the net
    serializers.save_hdf5(args.output, models[0])
    os.chmod(args.output, 0o777)
