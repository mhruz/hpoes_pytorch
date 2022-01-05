from .architectures import V2VModel as V2V, V2VModel88 as V2V88
from ..Utils import v2v_misc

import torch
import io
import numpy as np
import h5py
import argparse


# this function is from PyTorch > 1.7
def consume_prefix_in_state_dict_if_present(state_dict, prefix: str):
    r"""Strip the prefix in state_dict, if any.
    ..note::
        Given a `state_dict` from a DP/DDP model, a local model can load it by applying
        `consume_prefix_in_state_dict_if_present(state_dict, "module.")` before calling
        :meth:`torch.nn.Module.load_state_dict`.
    Args:
        state_dict (OrderedDict): a state-dict to be loaded to the model.
        prefix (str): prefix.
    """
    keys = sorted(state_dict.keys())
    for key in keys:
        if key.startswith(prefix):
            newkey = key[len(prefix) :]
            state_dict[newkey] = state_dict.pop(key)

    # also strip the prefix in metadata if any.
    if "_metadata" in state_dict:
        metadata = state_dict["_metadata"]
        for key in list(metadata.keys()):
            # for the metadata dict, the key can be:
            # '': for the DDP module, which we want to remove.
            # 'module': for the actual model.
            # 'module.xx.xx': for the rest.

            if len(key) == 0:
                continue
            newkey = key[len(prefix) :]
            metadata[newkey] = metadata.pop(key)


if __name__ == '__main__':
    # parse commandline
    parser = argparse.ArgumentParser(description='Test the V2V-PoseNet on Volumetric data.')
    parser.add_argument('test_h5', type=str, help='path to training h5 file with non-augmented data')
    parser.add_argument('net', type=str, help='path to the trained network (h5)')
    parser.add_argument('--batch_size', type=int, help='batch size in each iteration, default value = 8', default=8)
    parser.add_argument('--log', type=str, help='path to output log file')
    parser.add_argument('--data_label', type=str, help='the label of data in the H5 file, default = real_voxels',
                        default='real_voxels')
    parser.add_argument('--cubes_label', type=str, help='the label of cubes in the H5 file, default = cube',
                        default='cube')
    parser.add_argument('--compute_accuracy', action="store_true",
                        help='whether to compute the accuracy, if the labels are provided, default = False',
                        default=False)
    parser.add_argument('--read_data_to_memory', type=bool,
                        help='whether to read all the training data to memory, only '
                             'use for reasonable small data (< RAM)')
    parser.add_argument('output', type=str, help='name of the output file with predictions')
    args = parser.parse_args()

    f_test = h5py.File(args.test_h5, "r")

    if args.global_joints:
        num_joints = 1
    else:
        num_joints = f_test['labels'].shape[1]

    model = V2V88(1, num_joints)

    # make sure the model was saved by process with rank 0
    map_location = {'cuda:%d' % 0: 'cpu'}

    checkpoint = torch.load(args.net, map_location=map_location)
    consume_prefix_in_state_dict_if_present(checkpoint["model_state_dict"], prefix="module.")
    model.load_state_dict(checkpoint["model_state_dict"])

    epoch_idx = checkpoint["epoch"]
    loss = checkpoint["loss"]

    key = args.data_label
    batch_size = args.batch_size
    cubes_key = args.cubes_label

    # get the data
    if args.read_data_to_memory is not None and args.read_data_to_memory is True:
        num_samples = len(f_test[key])
        data_test = {cubes_key: f_test[cubes_key][:], 'labels': f_test['labels'][:], key: {}}

        for i in range(num_samples):
            data_test[key][str(i)] = f_test[key][str(i)][:]
    else:
        data_train = f_test

    data_test = f_test

    num_samples = len(data_test[key])

    for i in range(0, num_samples, batch_size):
        batch_data = []
        batch_labels = []
        cubes = []

        for idx in range(min(batch_size, num_samples - i)):
            pdata = data_test[key][str(i + idx)][:].tobytes()
            _file = io.BytesIO(pdata)
            data = np.load(_file)['arr_0']

            batch_data.append(data)
            if args.compute_accuracy:
                if "labels" not in data_test:
                    print("Cannot compute accuracy, since labels are not provided!")

                batch_labels.append(data_test['labels'][i + idx])

                yy, xx, zz = batch_labels[-1][:, 0], batch_labels[-1][:, 1], batch_labels[-1][:, 2]

                label_stack = v2v_misc.relative_cube_points3D2voxelcoord(batch_labels[-1])

        batch_data = np.expand_dims(batch_data, 1).astype(np.float32)
        batch_data = torch.tensor(batch_data)

        pred = model(batch_data)

