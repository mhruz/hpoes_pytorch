import math
import random
import numpy
import numpy as np
import torch
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import rotate
from scipy.interpolate import interpn


def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = numpy.asarray(axis)
    axis = axis / math.sqrt(numpy.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

    R =  numpy.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

    return R


def translation(ind3D, shift, Nvox=88):
    return ind3D + shift


def scale(ind3D, scale, Nvox=88):
    Nvox_half = Nvox / 2.0

    return (ind3D - Nvox_half) * scale + Nvox_half


def volumetric_scale(voxels, zoom_factor, threshold=0.4, poly_order=0):
    data_coords = numpy.zeros((3, 2), dtype=numpy.int)
    zoom_coords = numpy.zeros((3, 2), dtype=numpy.int)

    # axis fix for [row, col, depth] == [y, x, z]
    zoom_factor = [zoom_factor[1], zoom_factor[0], zoom_factor[2]]

    voxels = voxels.astype(np.float)
    zoom_data = zoom(voxels, zoom_factor, order=poly_order)
    zoom_data = zoom_data > threshold
    zoom_data_orig_shape = numpy.zeros_like(voxels)

    for i in range(len(zoom_data.shape)):
        if zoom_data.shape[i] < voxels.shape[i]:
            data_coords[i, 0] = (voxels.shape[i] - zoom_data.shape[i]) // 2
            data_coords[i, 1] = data_coords[i, 0] + zoom_data.shape[i]

            zoom_coords[i, 1] = zoom_data.shape[i]
        else:
            zoom_coords[i, 0] = (zoom_data.shape[i] - voxels.shape[i]) // 2
            zoom_coords[i, 1] = zoom_coords[i, 0] + voxels.shape[i]

            data_coords[i, 1] = voxels.shape[i]

    zoom_data_orig_shape[data_coords[0, 0]:data_coords[0, 1],
    data_coords[1, 0]:data_coords[1, 1],
    data_coords[2, 0]:data_coords[2, 1]] = zoom_data[zoom_coords[0, 0]:zoom_coords[0, 1],
                                           zoom_coords[1, 0]:zoom_coords[1, 1],
                                           zoom_coords[2, 0]:zoom_coords[2, 1]]

    return zoom_data_orig_shape


def rotation(ind3D, theta, Nvox):
    rot_mat = rotation_matrix((0, 0, 1), math.radians(-theta))
    Nvox_half = Nvox / 2.0
    for i, ind in enumerate(ind3D):
        ind = numpy.dot(rot_mat, ind - Nvox_half) + Nvox_half
        ind3D[i] = ind

    return ind3D


def rotation_volumetric(voxels, theta, threshold=0.4, poly_order=0):
    return rotate(voxels.astype(np.float), theta, reshape=False, order=poly_order) > threshold


def range_assert(ind3D, Nvox=88):
    _ind3D = numpy.asarray(numpy.rint(ind3D), dtype='int')

    return (ind3D[numpy.logical_and(numpy.max(_ind3D, 1) < Nvox, numpy.min(_ind3D, 1) > 0.)])


def voxelM(ind3d, Nvox=88):
    ind3d = numpy.asarray(numpy.rint(ind3d), dtype='int')
    ind3d = range_assert(ind3d, Nvox)
    ind3d = numpy.reshape(ind3d.T, (3, -1))

    V = numpy.zeros((Nvox, Nvox, Nvox), dtype='bool')
    # indexuje radek sloupec
    V[tuple(ind3d.tolist())] = True

    return V


def points3D2voxelcoord(points3D_centered, Nvox=88, cube=(250., 250., 250.)):
    cube = numpy.asarray(cube, 'float32')

    return float(Nvox) * (points3D_centered + (cube / 2.)) / cube


def relative_cube_points3D2voxelcoord(points3D_centered, Nvox=88):
    return float(Nvox) * (points3D_centered + 1.0) / 2.0


def voxelM2voxelcoord(V):
    voxelcoord = numpy.where(V == True)
    voxelcoord = list(zip(voxelcoord[0], voxelcoord[1], voxelcoord[2]))
    voxelcoord = numpy.asarray(voxelcoord)

    return voxelcoord


def augmentation_volumetric(volumetric_data, label_stack, cubes=None, grid_size_data=88, grid_size_label=44,
                            repetitions=1, scale_range=None, rotation_range=None, translation_range=None, app_thres=0.5,
                            poly_order=0):
    # handle default parameters
    if translation_range is None:
        translation_range = [-8.0, 8.0]
    if rotation_range is None:
        rotation_range = [-40.0, 40.0]
    if scale_range is None:
        scale_range = [0.8, 1.2]
    if cubes is None:
        cubes = [(250.0, 250.0, 250.0)] * len(volumetric_data)

    Vs = []
    label_stack_augs = []
    for i in range(repetitions):
        for volume_aug, label_stack_aug, cube in zip(volumetric_data, label_stack, cubes):
            #label_stack_aug = points3D2voxelcoord(label_stack_aug, Nvox=grid_size_label, cube=cube)
            label_stack_aug = relative_cube_points3D2voxelcoord(label_stack_aug, Nvox=grid_size_label)
            draw = [random.random(), random.random(), random.random()]
            if draw[0] >= app_thres:
                value = [random.uniform(scale_range[0], scale_range[1]),
                         random.uniform(scale_range[0], scale_range[1]),
                         random.uniform(scale_range[0], scale_range[1])]

                volume_aug = volumetric_scale(volume_aug, value, poly_order=poly_order)
                label_stack_aug = scale(label_stack_aug, value, grid_size_label)

            if draw[1] >= app_thres:
                value = random.uniform(rotation_range[0], rotation_range[1])
                value = -20
                volume_aug = rotation_volumetric(volume_aug, value, poly_order=0, threshold=0.4)
                label_stack_aug = rotation(label_stack_aug, value, grid_size_label)

            if draw[2] >= app_thres:
                value = [random.uniform(translation_range[0], translation_range[1]),
                         random.uniform(translation_range[0], translation_range[1]),
                         random.uniform(translation_range[0], translation_range[1])]

                ind3D_aug = voxelM2voxelcoord(volume_aug)
                if ind3D_aug.size != 0:
                    ind3D_aug = translation(ind3D_aug, (value[1], value[0], value[2]), grid_size_data)
                    volume_aug = voxelM(ind3D_aug, grid_size_data)

                    value[0] /= grid_size_data / float(grid_size_label)
                    value[1] /= grid_size_data / float(grid_size_label)
                    value[2] /= grid_size_data / float(grid_size_label)

                    label_stack_aug = translation(label_stack_aug, (value[0], value[1], value[2]), grid_size_label)

            V = volume_aug
            Vs.append(V)
            label_stack_augs.append(label_stack_aug)

    return numpy.asarray(Vs, dtype='float32'), numpy.asarray(label_stack_augs, dtype='float32')


def augmentation_volumetric_multichannel(ind3DList, labelStackList, cubes, Nvox_data=88, Nvox_label=44, repetitions=1,
                                         scale_range=None, rotation_range=None,
                                         translation_range=None, app_thres=0.5, poly_order=0):
    if translation_range is None:
        translation_range = [-8.0, 8.0]
    if rotation_range is None:
        rotation_range = [-40.0, 40.0]
    if scale_range is None:
        scale_range = [0.8, 1.2]

    Vs = []
    labelStack_augs = []
    # in this case the data have shape:
    # (number_of_samples, number_of_channels, data_X, data_Y, data_Z)
    number_of_channels = len(ind3DList[0])
    for volume_aug, labelStack, cube in zip(ind3DList, labelStackList, cubes):
        for i in range(repetitions):
            labelStack_aug = points3D2voxelcoord(labelStack, Nvox=Nvox_label, cube=cube)
            draw = [random.random(), random.random(), random.random()]
            # print draw
            if draw[0] >= app_thres:
                value = [random.uniform(scale_range[0], scale_range[1]),
                         random.uniform(scale_range[0], scale_range[1]),
                         random.uniform(scale_range[0], scale_range[1])]

                for channel in range(number_of_channels):
                    volume_aug[channel] = volumetric_scale(volume_aug[channel], value, poly_order=poly_order)

                labelStack_aug = scale(labelStack_aug, value, Nvox_label)

            if draw[1] >= app_thres:
                value = random.uniform(rotation_range[0], rotation_range[1])
                for channel in range(number_of_channels):
                    volume_aug[channel] = rotation_volumetric(volume_aug[channel], value, poly_order=poly_order)

                labelStack_aug = rotation(labelStack_aug, value, Nvox_label)
            if draw[2] >= app_thres:
                value = [random.uniform(translation_range[0], translation_range[1]),
                         random.uniform(translation_range[0], translation_range[1]),
                         random.uniform(translation_range[0], translation_range[1])]
                for channel in range(number_of_channels):
                    volume_aug[channel] = voxelM2voxelcoord(volume_aug[channel])
                    volume_aug[channel] = translation(volume_aug[channel], (value[0], value[1], value[2]), Nvox_data)
                    volume_aug[channel] = voxelM(volume_aug[channel], Nvox_data)

                value[0] /= Nvox_data / float(Nvox_label)
                value[1] /= Nvox_data / float(Nvox_label)
                value[2] /= Nvox_data / float(Nvox_label)

                labelStack_aug = translation(labelStack_aug, (value[0], value[1], value[2]), Nvox_label)

            V = numpy.zeros((number_of_channels, Nvox_data, Nvox_data, Nvox_data), dtype=numpy.float32)
            for channel in range(number_of_channels):
                V[channel][:, :, :] = volume_aug[channel]

            Vs.append(V)
            labelStack_augs.append(labelStack_aug)

    return numpy.asarray(Vs, dtype='float32'), numpy.asarray(labelStack_augs, dtype='float32')


def augmentation_volumetric_full_voxel(ind3DList, labelStackList, cubes, Nvox_data=88, Nvox_label=44, repetitions=1,
                                       scale_range=None, rotation_range=None,
                                       translation_range=None, app_thres=0.5, poly_order=0):
    if translation_range is None:
        translation_range = [-8.0, 8.0]
    if rotation_range is None:
        rotation_range = [-40.0, 40.0]
    if scale_range is None:
        scale_range = [0.8, 1.2]

    Vs = []
    labelStack_augs = []
    for volume_aug, labelStack, cube in zip(ind3DList, labelStackList, cubes):
        for i in range(repetitions):
            labelStack_aug = points3D2voxelcoord(labelStack, Nvox=Nvox_label, cube=cube)
            draw = [random.random(), random.random(), random.random()]
            if draw[0] >= app_thres:
                value = [random.uniform(scale_range[0], scale_range[1]),
                         random.uniform(scale_range[0], scale_range[1]),
                         random.uniform(scale_range[0], scale_range[1])]

                volume_aug = volumetric_scale(volume_aug, value, poly_order=poly_order, threshold=0.6)

                labelStack_aug = scale(labelStack_aug, value, Nvox_label)
            if draw[1] >= app_thres:
                value = random.uniform(rotation_range[0], rotation_range[1])
                volume_aug = rotation_volumetric(volume_aug, value, poly_order=poly_order, threshold=0.6)

                labelStack_aug = rotation(labelStack_aug, value, Nvox_label)
            if draw[2] >= app_thres:
                value = [random.uniform(translation_range[0], translation_range[1]),
                         random.uniform(translation_range[0], translation_range[1]),
                         random.uniform(translation_range[0], translation_range[1])]

                ind3D_aug = voxelM2voxelcoord(volume_aug)
                ind3D_aug = translation(ind3D_aug, (value[0], value[1], value[2]), Nvox_data)
                volume_aug = voxelM(ind3D_aug, Nvox_data)

                value[0] /= Nvox_data / float(Nvox_label)
                value[1] /= Nvox_data / float(Nvox_label)
                value[2] /= Nvox_data / float(Nvox_label)

                labelStack_aug = translation(labelStack_aug, (value[0], value[1], value[2]), Nvox_label)

            V = volume_aug
            Vs.append(V)
            labelStack_augs.append(labelStack_aug)

    return numpy.asarray(Vs, dtype='float32'), numpy.asarray(labelStack_augs, dtype='float32')


# promitne fullvoxel data do dane osy
# vstup data je 3D boolean matrix
def project_fullvoxel(data, axis=2, flip=False):
    if flip:
        data = numpy.flip(data, axis=axis)
    mask_proj = data.argmax(axis=axis)
    idx_proj = numpy.where(mask_proj != 0)
    idx_proj3 = list(idx_proj)
    idx_proj3.insert(axis, mask_proj[idx_proj])
    data[:] = False
    data[idx_proj3] = True
    if flip:
        data = numpy.flip(data, axis=axis)
    return data


def make_heat_maps_gpu(label_stack, sigma=1.7, grid_size=44, nominal_cube_shape=250, cubes=None, device=0):
    # handle device string
    device = "cuda:{}".format(device)
    # select proper mem size for gridsize
    if grid_size <= 255:
        grid_size_dtype = torch.uint8
    else:
        grid_size_dtype = torch.int16

    batch_size = label_stack.shape[0]
    num_points = label_stack.shape[1]

    label_stack = numpy.moveaxis(label_stack, 2, 0)
    label_stack = label_stack.reshape([3, batch_size, num_points, 1, 1, 1])

    inv_radius = torch.ones(batch_size, dtype=torch.float32, device=device)

    if cubes is not None and not all(cubes[:, 0] == nominal_cube_shape):
        inv_radius /= (nominal_cube_shape / cubes[:, 0]) ** 2

    inv_radius *= -1 / (2 * sigma * sigma)

    x = torch.arange(grid_size, dtype=grid_size_dtype, device=device).reshape((1, 1, grid_size, 1, 1))
    y = torch.arange(grid_size, dtype=grid_size_dtype, device=device).reshape((1, 1, 1, grid_size, 1))
    z = torch.arange(grid_size, dtype=grid_size_dtype, device=device).reshape((1, 1, 1, 1, grid_size))

    points = torch.from_numpy(label_stack).to(device)

    dx = points[0] - x
    dy = points[1] - y
    dz = points[2] - z
    dx = dx ** 2
    dy = dy ** 2
    dz = dz ** 2
    ds = (dx + dy + dz)

    inv_radius = inv_radius.expand(ds.shape[1], ds.shape[2], ds.shape[3], ds.shape[4], ds.shape[0])
    inv_radius = inv_radius.permute(4, 0, 1, 2, 3)

    value = torch.exp(inv_radius * ds)

    return value


def make_global_heat_map_gpu(label_stack, sigma=1.7, grid_size=44, nominal_cube_shape=250, cubes=None, device=0):
    """
    The function generates one heatmap for all joints. This is useful when we want to estimate the locations of the
    joints without classifying them.
    Args:
        label_stack:
        sigma:
        grid_size:
        nominal_cube_shape:
        cubes:
        device:

    Returns:

    """
    # handle device string
    device = "cuda:{}".format(device)
    # select proper mem size for gridsize
    if grid_size <= 255:
        grid_size_dtype = torch.uint8
    else:
        grid_size_dtype = torch.int16

    batch_size = label_stack.shape[0]
    num_points = label_stack.shape[1]

    label_stack = numpy.moveaxis(label_stack, 2, 0)
    label_stack = label_stack.reshape([3, batch_size, num_points, 1, 1, 1])

    inv_radius = torch.ones(batch_size, dtype=torch.float32, device=device)

    if cubes is not None and not all(cubes[:, 0] == nominal_cube_shape):
        inv_radius /= (nominal_cube_shape / cubes[:, 0]) ** 2

    inv_radius *= -1 / (2 * sigma * sigma)

    x = torch.arange(grid_size, dtype=grid_size_dtype, device=device).reshape((1, 1, grid_size, 1, 1))
    y = torch.arange(grid_size, dtype=grid_size_dtype, device=device).reshape((1, 1, 1, grid_size, 1))
    z = torch.arange(grid_size, dtype=grid_size_dtype, device=device).reshape((1, 1, 1, 1, grid_size))

    points = torch.from_numpy(label_stack).to(device)

    dx = points[0] - x
    dy = points[1] - y
    dz = points[2] - z
    dx = dx ** 2
    dy = dy ** 2
    dz = dz ** 2
    ds = (dx + dy + dz)

    inv_radius = inv_radius.expand(ds.shape[1], ds.shape[2], ds.shape[3], ds.shape[4], ds.shape[0])
    inv_radius = inv_radius.permute(4, 0, 1, 2, 3)

    value = torch.exp(inv_radius * ds)
    value, indices = torch.max(value, dim=1, keepdim=True)

    return value


def heatMaps2VoxelCPU(heatmaps, Nvox=44):
    if len(heatmaps.shape) < 5:
        heatmaps = numpy.expand_dims(heatmaps, 0)

    ret = numpy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=numpy.float32)

    grid_size = Nvox
    x = numpy.arange(grid_size, dtype=numpy.float32).reshape([1, grid_size, 1, 1])
    y = numpy.arange(grid_size, dtype=numpy.float32).reshape([1, 1, grid_size, 1])
    z = numpy.arange(grid_size, dtype=numpy.float32).reshape([1, 1, 1, grid_size])

    for (i, hm) in enumerate(heatmaps):
        w_sum = numpy.sum(hm, axis=(1, 2, 3))
        avg_x = numpy.sum(x * hm, axis=(1, 2, 3)) / w_sum
        avg_y = numpy.sum(y * hm, axis=(1, 2, 3)) / w_sum
        avg_z = numpy.sum(z * hm, axis=(1, 2, 3)) / w_sum

        p3D = numpy.array([avg_x, avg_y, avg_z], dtype=numpy.float32).T
        ret[i] = p3D

    return ret


def heatMaps2Points3D_smooth_max(heatmaps, cubes, Nvox=44, smooth_size=3):
    # heatmaps have shape (batch, num_joints, Nvox, Nvox, Nvox)
    if len(heatmaps.shape) < 5:
        heatmaps = numpy.expand_dims(heatmaps, 0)

    # smooth_size needs to be odd
    if smooth_size % 2 == 0:
        smooth_size += 1

    ret = numpy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=numpy.float32)
    ret_conf = numpy.zeros((heatmaps.shape[0], heatmaps.shape[1]), dtype=numpy.float32)

    x1 = numpy.arange(smooth_size)
    x2 = numpy.arange(smooth_size)
    x3 = numpy.arange(smooth_size)

    # go through batch data
    for (i, hm) in enumerate(heatmaps):
        # go through joints
        for (j, h) in enumerate(hm):
            ind = numpy.array(numpy.unravel_index(numpy.argmax(h, axis=None), h.shape))
            # send the local submatrix to compute the weighted average
            # H = h[ind[0]-1:ind[0]+2, ind[1]-1:ind[1]+2, ind[2]-1:ind[2]+2]

            H = numpy.zeros((smooth_size, smooth_size, smooth_size), dtype=numpy.float32)
            left_bound = -(smooth_size - 1) // 2
            right_bound = (smooth_size - 1) // 2 + 1
            center = (smooth_size - 1) // 2
            for a in range(left_bound, right_bound):
                for b in range(left_bound, right_bound):
                    for c in range(left_bound, right_bound):
                        if ind[0] + a >= 0 and ind[0] + a < Nvox and ind[1] + b >= 0 and ind[1] + b < Nvox and ind[
                            2] + c >= 0 and ind[2] + c < Nvox:
                            H[a - left_bound, b - left_bound, c - left_bound] = h[
                                int(ind[0]) + a, int(ind[1]) + b, int(ind[2]) + c]
                        else:
                            H[a - left_bound, b - left_bound, c - left_bound] = 0

            H = numpy.expand_dims(H, 0)
            ind_s = heatMaps2VoxelCPU(H, Nvox=smooth_size)[0]
            ret[i, j, :] = ((ind + ind_s - center) / float(Nvox) - 0.5) * cubes[i][0]

            ret_conf[i, j] = interpn((x1, x2, x3), H[0], ind_s, bounds_error=False, fill_value=None)

    return ret, ret_conf


def detect_best_joint_locations(heatmaps, n_joints, suppress_size):
    # heatmaps have shape (batch, num_joints, Nvox, Nvox, Nvox)
    if len(heatmaps.shape) < 5:
        heatmaps = numpy.expand_dims(heatmaps, 0)

    # pad the heatmaps
    padded_heatmaps = np.zeros((heatmaps.shape[0], heatmaps.shape[1], heatmaps.shape[2] + suppress_size - 1,
                                heatmaps.shape[3] + suppress_size - 1, heatmaps.shape[4] + suppress_size - 1))

    pad_start = (suppress_size - 1) // 2

    padded_heatmaps[:, :, pad_start:-pad_start, pad_start:-pad_start, pad_start:-pad_start] = heatmaps

    size = padded_heatmaps.shape[2]
    # H = numpy.zeros((suppress_size, suppress_size, suppress_size), dtype=numpy.float32)

    maximums_batch = []

    # go through batch data
    for (i, hm) in enumerate(padded_heatmaps):
        maximums = []
        values = []
        for a in range(size):
            for b in range(size):
                for c in range(size):
                    H = hm[0][a:a + suppress_size, b:b + suppress_size, c:c + suppress_size]
                    max_idx = np.unravel_index(np.argmax(H), H.shape)
                    if all(idx == pad_start for idx in max_idx):
                        values.append(H[pad_start, pad_start, pad_start])
                        maximums.append([a + pad_start, b + pad_start, c + pad_start])

        maxs_sorted = [x for _, x in sorted(zip(values, maximums), reverse=True)]
        maximums_batch.append(maxs_sorted[:n_joints])

    return maximums_batch
