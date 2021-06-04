from .architectures import V2VModel as V2V
from ..Utils import v2v_misc
import h5py
import numpy as np
import argparse
import time
import io
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda import amp
import torch.distributed as dist


def train_net_on_node(local_rank, global_rank_offset, world_size, gpu_rank, args):
    # global rank of the process
    rank = local_rank + global_rank_offset

    # device string representation
    device = "cuda:{}".format(gpu_rank)

    augment = True
    logging = False
    dev = False

    if args.log is not None:
        log_filename, log_extension = os.path.splitext(args.log)
        args.log = args.log[len(log_extension)]

        f_log = open("{}_rank_{}{}".format(log_filename, rank, log_extension), 'wt')
        logging = True

    batch_size = args.batch_size
    max_epochs = args.max_epoch
    epoch_idx = 0
    loss = None

    # get the closest approximation of local batch size
    local_batch_size = batch_size // world_size
    # compute the remainder after each rank receives local batch size
    batch_remainder = local_batch_size % world_size
    # divide the remainder into the first N ranks, where N = remainder
    if rank < batch_remainder:
        local_batch_size += 1

    # compute the starting index of data for this rank
    # if data are split e.g. |3|3|2|2| => batch size = 10, number of ranks = 4
    # local_batch_start_idx will be:
    # rank0 = 0; rank1 = 3; rank2 = 6; rank3 = 8
    local_batch_start_idx = 0
    for r in range(rank):
        local_batch_start_idx += local_batch_size
        if r < batch_remainder:
            local_batch_start_idx += 1

    f_train = h5py.File(args.train_h5, "r")
    # determine the number of joints
    num_joints = f_train['labels'].shape[1]

    if world_size > 1:
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
        # create multiple models on multiple GPUs
        model = DDP(V2V(1, num_joints).to(gpu_rank), device_ids=[gpu_rank])
    else:
        model = V2V(1, num_joints).to(gpu_rank)
    # Choose an optimizer algorithm
    optimizer = optim.RMSprop(model.parameters(), lr=0.00025)
    # choose criterion
    loss_fn = nn.MSELoss()

    if args.init_net is not None:
        # make sure the model was saved by process with rank 0
        map_location = {'cuda:%d' % 0: 'cuda:%d' % gpu_rank}

        checkpoint = torch.load(args.init_net, map_location=map_location)
        model.load_state_dict(checkpoint["model_state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        epoch_idx = checkpoint["epoch"]
        loss = checkpoint["loss"]

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
        data_train = {'cubes': f_train['cubes'][:], 'labels': f_train['labels'][:], key: {}}

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
            data_dev = {'cubes': f_dev['cubes'][:], 'labels': f_dev['labels'][:], key: {}}

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
    indexes_all = np.asarray(list(range(len(data_train[key].keys()))))
    # the tensor for shuffled indexes, since we use NCCL backend, it has to be stored on GPU
    indexes_all_tensor = torch.from_numpy(indexes_all).to(device)

    num_samples = len(data_train[key])

    while epoch_idx < max_epochs:
        if logging:
            f_log.write('Epoch {}\n'.format(epoch_idx))

        # compute the loss on dev data if available
        if dev:
            dev_loss = []
            num_samples = len(data_dev[key])

            model.eval()

            # i represents the starting index of global batch
            for i in range(0, num_samples, batch_size):
                batch_data = []
                batch_labels = []

                # compute the index of data for this rank
                idx0 = i + local_batch_start_idx

                for idx in range(min(local_batch_size, num_samples - idx0)):
                    pdata = data_dev[key][str(idx0 + idx)][:].tostring()
                    _file = io.BytesIO(pdata)
                    data = np.load(_file)['arr_0']

                    batch_data.append(data)
                    batch_labels.append(data_dev['labels'][idx0 + idx])

                batch_data = np.array(batch_data, dtype=np.float32)
                batch_labels = np.array(batch_labels)
                batch_data = np.expand_dims(batch_data, 1)

                target_gpu = v2v_misc.make_heat_maps_gpu(batch_labels, device=gpu_rank)
                batch_data_gpu = torch.from_numpy(batch_data).to(device)

                pred = model(batch_data_gpu)
                loss = loss_fn(pred, target_gpu)

                dev_loss.append(loss.item())

            dev_loss = np.mean(np.array(dev_loss))
            f_log.write("Dev loss on GPU {}/{}: {}\n".format(rank, device, dev_loss))
            f_log.flush()

        # in every epoch shuffle the indexes of data
        # rank 0 will shuffle
        if rank == 0:
            np.random.shuffle(indexes_all)
            indexes_all_tensor = torch.from_numpy(indexes_all).to(device)
            if logging:
                f_log.write("Shuffled data. Preparing for broadcast.")

        if world_size > 1:
            torch.distributed.broadcast(indexes_all_tensor, 0)

        if logging:
            if rank == 0:
                f_log.write("Data broadcasted.")
            else:
                f_log.write("Data received.")

        model.train()

        for i in range(0, num_samples, batch_size):
            batch_data = []
            batch_labels = []
            cubes = []

            # compute the index of data for this rank
            idx0 = i + local_batch_start_idx

            # if there is not enough data for computation, end the epoch prematurely
            if num_samples - i < batch_size:
                break

            for idx in range(min(local_batch_size, num_samples - idx0)):
                index_of_data = indexes_all_tensor[idx0 + idx].item()

                pdata = data_train[key][str(index_of_data)][:].tostring()
                _file = io.BytesIO(pdata)
                data = np.load(_file)['arr_0']

                batch_data.append(data)
                batch_labels.append(data_train['labels'][index_of_data])

                cubes.append(data_train['cubes'][index_of_data])

            (batch_data, batch_labels) = v2v_misc.augmentation_volumetric(batch_data, batch_labels, cubes)
            batch_data = np.expand_dims(batch_data, 1)

            target_gpu = v2v_misc.make_heat_maps_gpu(batch_labels, device=gpu_rank)
            batch_data_gpu = torch.from_numpy(batch_data).to(device)

            optimizer.zero_grad()

            pred = model(batch_data_gpu)
            loss = loss_fn(pred, target_gpu)

            if logging:
                f_log.write('Iteration {}, loss on GPU {}: {}\n'.format(i//batch_size, rank, loss.item()))

            loss.backward()

            if logging:
                f_log.write('Loss backward successful.\n')

            optimizer.step()

            if logging:
                f_log.write('Optimizer step successful.\n')
                f_log.flush()

        epoch_idx += 1

        # save the intermediate net
        if rank == 0 and epoch_idx % args.save_iter == 0:
            filename = args.output + '_epoch_' + str(epoch_idx) + '.tar'
            if args.log:
                f_log.write("Saving intermediate model to: {}\n".format(filename))
                f_log.flush()

            torch.save({
                'epoch': epoch_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item()
            }, filename)
            os.chmod(filename, 0o777)

            if args.log:
                f_log.write("Model saved successfully\n")
                f_log.flush()

        # wait for the model to save
        dist.barrier()

    if logging:
        f_log.close()

    # save the net
    if rank == 0:
        torch.save({
            'epoch': epoch_idx,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item()
        }, args.output)

        os.chmod(args.output, 0o777)


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

    # when training on multi-node environment, make sure these environment variables were set before running script
    try:
        gpus_per_node = int(os.environ['PBS_NGPUS'])
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        node_id = int(os.environ['OMPI_COMM_WORLD_RANK'])
        gpu_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    except KeyError:
        gpus_per_node = 1
        world_size = 1
        node_id = 0
        gpu_rank = 0

    print("gpus_per_node: {}".format(gpus_per_node))
    print("gpu_rank: {}".format(gpu_rank))
    print("size: {}".format(world_size))
    print("node_id: {}".format(node_id))

    mp.spawn(train_net_on_node, args=(node_id, world_size, gpu_rank, args))


