import sys

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
            newkey = key[len(prefix):]
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
            newkey = key[len(prefix):]
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
    parser.add_argument('--global_joints', action="store_true",
                        help='whether to predict the identity of the joints (False) or whether to predict one heatmap'
                             ' of unknown joint locations (True), default = False',
                        default=False)
    parser.add_argument('--compute_accuracy', action="store_true",
                        help='whether to compute the accuracy, if the labels are provided, default = False',
                        default=False)
    parser.add_argument('--read_data_to_memory', type=bool,
                        help='whether to read all the training data to memory, only '
                             'use for reasonable small data (< RAM)')
    parser.add_argument('output', type=str, help='name of the output file with predictions')
    args = parser.parse_args()

    f_test = h5py.File(args.test_h5, "r")
    f_output = open(args.output, "w")

    if args.global_joints:
        num_joints = 1
        model = V2V88(1, num_joints)
    else:
        num_joints = f_test['labels'].shape[1]
        model = V2V(1, num_joints)

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

    acc = 0.0

    # get the data
    if args.read_data_to_memory is not None and args.read_data_to_memory is True:
        num_samples = len(f_test[key])
        data_test = {cubes_key: f_test[cubes_key][:]}
        if args.compute_accuracy:
            try:
                data_test['labels'] = f_test['labels'][:]
            except KeyError:
                print("When compute_accuracy is set, you need to provide labels in th H5 file.")
                sys.exit(1)

        for i in range(num_samples):
            data_test[key][str(i)] = f_test[key][str(i)][:]
    else:
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

            cubes.append(data_test[cubes_key][i + idx])

            batch_data.append(data)
            if args.compute_accuracy:
                if "labels" not in data_test:
                    print("Cannot compute accuracy, since labels are not provided!")

                batch_labels.append(data_test['labels'][i + idx])

        batch_data = np.expand_dims(batch_data, 1).astype(np.float32)
        batch_data = torch.tensor(batch_data)

        pred = model(batch_data)

        points_3d = v2v_misc.heat_maps2points3d_smooth_max(pred.detach().numpy(), cubes)

        for out in points_3d[0]:
            f_output.write(" ".join(map(str, out.flatten())))
            f_output.write("\n")

        f_output.flush()

        if args.compute_accuracy:
            labels = []
            for label, cube in zip(batch_labels, cubes):
                labels.append(label * np.tile(cube, (label.shape[0], 1)))

            acc_batch = np.array(labels) - points_3d[0]
            acc_batch = np.square(acc_batch)
            acc_batch = np.sum(acc_batch, axis=2)
            acc_batch = np.sqrt(acc_batch)
            acc_batch = np.mean(acc_batch)

            acc += acc_batch / num_samples

    f_output.write("Accuracy: {}".format(acc))
    f_output.close()