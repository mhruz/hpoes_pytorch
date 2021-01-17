import sys
import math
import random
import time
import numpy
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

    return numpy.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                        [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                        [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def translation(ind3D, shift, Nvox=88):
    return ind3D + shift


def scale(ind3D, scale, Nvox=88):
    Nvox_half = Nvox / 2.0

    return (ind3D - Nvox_half) * scale + Nvox_half


def volumetric_scale(voxels, zoom_factor, threshold=0.4, poly_order=0):
    data_coords = numpy.zeros((3, 2), dtype=numpy.int)
    zoom_coords = numpy.zeros((3, 2), dtype=numpy.int)

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
    rot_mat = rotation_matrix((0, 0, 1), math.radians(theta))
    Nvox_half = Nvox / 2.0
    for i, ind in enumerate(ind3D):
        ind = numpy.dot(rot_mat, ind - Nvox_half) + Nvox_half
        ind3D[i] = ind

    return ind3D


def rotation_volumetric(voxels, theta, threshold=0.4, poly_order=0):
    return rotate(voxels, theta, reshape=False, order=poly_order) > threshold


def range_assert(ind3D, Nvox=88):
    _ind3D = numpy.asarray(numpy.rint(ind3D), dtype='int')

    return (ind3D[numpy.logical_and(numpy.max(_ind3D, 1) < Nvox, numpy.min(_ind3D, 1) > 0.)])


def voxelM(ind3d, Nvox=88):
    ind3d = numpy.asarray(numpy.rint(ind3d), dtype='int')
    ind3d = range_assert(ind3d, Nvox)
    ind3d = numpy.reshape(ind3d.T, (3, -1))

    V = numpy.zeros((Nvox, Nvox, Nvox), dtype='bool')
    # indexuje radek sloupec
    V[ind3d.tolist()] = True

    return V


def points3D2voxelcoord(points3D_centered, Nvox=88, cube=(250., 250., 250.)):
    cube = numpy.asarray(cube, 'float32')

    return float(Nvox) * (points3D_centered + (cube / 2.)) / cube


def voxelM2voxelcoord(V):
    voxelcoord = numpy.where(V == True)
    voxelcoord = list(zip(voxelcoord[0], voxelcoord[1], voxelcoord[2]))
    voxelcoord = numpy.asarray(voxelcoord)

    return voxelcoord


def augmentation_volumetric(volumetric_data, label_stack, cubes, grid_size_data=88, grid_size_label=44, repetitions=1,
                            scale_range=None, rotation_range=None, translation_range=None, app_thres=0.5, poly_order=0):
    # handle default parameters
    if translation_range is None:
        translation_range = [-8.0, 8.0]
    if rotation_range is None:
        rotation_range = [-40.0, 40.0]
    if scale_range is None:
        scale_range = [0.8, 1.2]

    Vs = []
    label_stack_aug = []
    for volume_aug, labelStack, cube in zip(volumetric_data, label_stack, cubes):
        for i in range(repetitions):
            label_stack_aug = points3D2voxelcoord(labelStack, Nvox=grid_size_label, cube=cube)
            draw = [random.random(), random.random(), random.random()]
            if draw[0] >= app_thres:
                value = [random.uniform(scale_range[0], scale_range[1]),
                         random.uniform(scale_range[0], scale_range[1]),
                         random.uniform(scale_range[0], scale_range[1])]

                volume_aug = volumetric_scale(volume_aug, value, poly_order=poly_order)

                label_stack_aug = scale(label_stack_aug, value, grid_size_label)
            if draw[1] >= app_thres:
                value = random.uniform(rotation_range[0], rotation_range[1])
                volume_aug = rotation_volumetric(volume_aug, value, poly_order=poly_order)

                label_stack_aug = rotation(label_stack_aug, value, grid_size_label)
            if draw[2] >= app_thres:
                value = [random.uniform(translation_range[0], translation_range[1]),
                         random.uniform(translation_range[0], translation_range[1]),
                         random.uniform(translation_range[0], translation_range[1])]

                ind3D_aug = voxelM2voxelcoord(volume_aug)
                if ind3D_aug.size != 0:
                    ind3D_aug = translation(ind3D_aug, (value[0], value[1], value[2]), grid_size_data)
                    volume_aug = voxelM(ind3D_aug, grid_size_data)

                    value[0] /= grid_size_data / float(grid_size_label)
                    value[1] /= grid_size_data / float(grid_size_label)
                    value[2] /= grid_size_data / float(grid_size_label)

                    label_stack_aug = translation(label_stack_aug, (value[0], value[1], value[2]), grid_size_label)

            V = volume_aug
            Vs.append(V)
            label_stack_aug.append(label_stack_aug)

    return numpy.asarray(Vs, dtype='float32'), numpy.asarray(label_stack_aug, dtype='float32')


def augmentation_volumetric2(ind3DList, labelStackList, cubes, egos, batch_size, Nvox_data=88, Nvox_label=44,
                             repetitions=1, scale_range=[0.8, 1.2], rotation_range=[-40.0, 40.0],
                             translation_range=[-8.0, 8.0], app_thres=0.5, poly_order=0):
    Vs = []
    labelStack_augs = []
    for volume_aug, labelStack, cube, ego in zip(ind3DList, labelStackList, cubes, egos):
        repetitions = 1
        if ego:
            repetitions = 3
        for i in range(repetitions):
            labelStack_aug = points3D2voxelcoord(labelStack, Nvox=Nvox_label, cube=cube)
            draw = [random.random(), random.random(), random.random()]
            if draw[0] >= app_thres:
                value = [random.uniform(scale_range[0], scale_range[1]),
                         random.uniform(scale_range[0], scale_range[1]),
                         random.uniform(scale_range[0], scale_range[1])]

                volume_aug = volumetric_scale(volume_aug, value, poly_order=poly_order)

                labelStack_aug = scale(labelStack_aug, value, Nvox_label)
            if draw[1] >= app_thres:
                value = random.uniform(rotation_range[0], rotation_range[1])
                volume_aug = rotation_volumetric(volume_aug, value, poly_order=poly_order)

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
            if len(Vs) == batch_size:
                return numpy.asarray(Vs, dtype='float32'), numpy.asarray(labelStack_augs, dtype='float32')

    return numpy.asarray(Vs, dtype='float32'), numpy.asarray(labelStack_augs, dtype='float32')


def augmentation_volumetric16(ind3DList, labelStackList, cubes, Nvox_data=88, Nvox_label=44, repetitions=1,
                              scale_range=[0.8, 1.2], rotation_range=[-40.0, 40.0], translation_range=[-8.0, 8.0],
                              app_thres=0.5, poly_order=0):
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

                volume_aug = volumetric_scale(volume_aug, value, poly_order=poly_order)

                labelStack_aug = scale(labelStack_aug, value, Nvox_label)
            if draw[1] >= app_thres:
                value = random.uniform(rotation_range[0], rotation_range[1])
                volume_aug = rotation_volumetric(volume_aug, value, poly_order=poly_order)

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

    return numpy.asarray(Vs, dtype='float16'), numpy.asarray(labelStack_augs, dtype='float16')


def augmentation_volumetric_multichannel(ind3DList, labelStackList, cubes, Nvox_data=88, Nvox_label=44, repetitions=1,
                                         scale_range=[0.8, 1.2], rotation_range=[-40.0, 40.0],
                                         translation_range=[-8.0, 8.0], app_thres=0.5, poly_order=0):
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
                                       scale_range=[0.8, 1.2], rotation_range=[-40.0, 40.0],
                                       translation_range=[-8.0, 8.0], app_thres=0.5, poly_order=0):
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

                volume_aug = volumetric_scale(volume_aug, value, poly_order=poly_order)

                labelStack_aug = scale(labelStack_aug, value, Nvox_label)
            if draw[1] >= app_thres:
                value = random.uniform(rotation_range[0], rotation_range[1])
                volume_aug = rotation_volumetric(volume_aug, value, poly_order=poly_order)

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


# prevede labels na voxels
def makeHeatMaps(labelStackArray, sigma=1.7, Nvox=44, threads=4):
    s = time.time()
    global indLall
    global indLallmatrix

    E = numpy.zeros((Nvox, Nvox, Nvox), dtype='float32')
    indLall = numpy.where(E == 0)
    indLallmatrix = numpy.asarray(indLall, dtype='float32')
    # prohodit abych mel x y z
    indLallmatrix = numpy.flip(indLallmatrix, 0)
    # udelej z [[x x x x] [y y y y] [z z z y] ]  ->  [ [x y z] [x y z] ...
    indLallmatrix = numpy.swapaxes(indLallmatrix, 0, 1)
    heatMapArray = []
    p = Pool(threads)
    for labelStack in labelStackArray:
        param_list = [(joint, sigma, Nvox) for joint in labelStack]
        heatMap = p.map(makeHeatMap, param_list)
        # heatMap = makeHeatMap(param_list[0])
        heatMapArray.append(heatMap)

    heatMapArray = numpy.asarray(heatMapArray)
    # print heatMapArray.shape
    heatMapArray = numpy.moveaxis(heatMapArray, 1, -1)
    print(time.time() - s)

    return heatMapArray


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


def makeHeatMapsGPU_linear(labelStackArray, sigma=3.0, Nvox=44, nominal_cube_shape=250, cubes=None, device=0):
    with cupy.cuda.Device(device):
        if cubes is not None:
            cubes = cupy.asarray(cubes)

        batch_size = labelStackArray.shape[0]
        grid_size = Nvox
        num_points = labelStackArray.shape[1]
        labelStackArray = numpy.moveaxis(labelStackArray, 2, 0)
        labelStackArray = labelStackArray.reshape([3, batch_size, num_points, 1, 1, 1])

        inv_radius = cupy.ones((batch_size), dtype=cupy.float32)

        if cubes is not None and not all(cubes[:, 0] == nominal_cube_shape):
            inv_radius /= (nominal_cube_shape / cubes[:, 0])

        x = cupy.arange(grid_size, dtype=cupy.float32).reshape([1, 1, grid_size, 1, 1])
        y = cupy.arange(grid_size, dtype=cupy.float32).reshape([1, 1, 1, grid_size, 1])
        z = cupy.arange(grid_size, dtype=cupy.float32).reshape([1, 1, 1, 1, grid_size])

        points = cupy.array(labelStackArray)

        dx = abs(points[0] - x)
        dy = abs(points[1] - y)
        dz = abs(points[2] - z)
        ds = (dx + dy + dz)

        inv_radius = F.broadcast_to(inv_radius, (ds.shape[1], ds.shape[2], ds.shape[3], ds.shape[4], ds.shape[0]))
        inv_radius = inv_radius.data
        inv_radius = cupy.moveaxis(inv_radius, 4, 0)

        value = numpy.maximum(0, 1 - (ds / sigma) * inv_radius)

    return value


def heatMaps2Points3DGPU(heatmaps, Nvox=44, cube=(250, 250, 250)):
    if len(heatmaps.shape) < 5:
        heatmaps = cupy.expand_dims(heatmaps, 0)

    ret = cupy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=numpy.float32)

    grid_size = Nvox
    x = cupy.arange(grid_size, dtype=cupy.float32).reshape([1, grid_size, 1, 1])
    y = cupy.arange(grid_size, dtype=cupy.float32).reshape([1, 1, grid_size, 1])
    z = cupy.arange(grid_size, dtype=cupy.float32).reshape([1, 1, 1, grid_size])

    for (i, hm) in enumerate(heatmaps):
        w_sum = cupy.sum(hm, axis=[1, 2, 3])

        avg_x = cupy.sum(x * hm, axis=[1, 2, 3]) / w_sum
        avg_y = cupy.sum(y * hm, axis=[1, 2, 3]) / w_sum
        avg_z = cupy.sum(z * hm, axis=[1, 2, 3]) / w_sum

        avg_x = (avg_x / float(Nvox) - 0.5) * cube[0]
        avg_y = (avg_y / float(Nvox) - 0.5) * cube[1]
        avg_z = (avg_z / float(Nvox) - 0.5) * cube[2]

        p3D = cupy.array([avg_x, avg_y, avg_z], dtype=numpy.float32).T
        ret[i] = p3D

    return ret


def heatMaps2VoxelGPU(heatmaps, Nvox=44):
    if len(heatmaps.shape) < 5:
        heatmaps = cupy.expand_dims(heatmaps, 0)

    ret = cupy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=numpy.float32)

    grid_size = Nvox
    x = cupy.arange(grid_size, dtype=cupy.float32).reshape([1, grid_size, 1, 1])
    y = cupy.arange(grid_size, dtype=cupy.float32).reshape([1, 1, grid_size, 1])
    z = cupy.arange(grid_size, dtype=cupy.float32).reshape([1, 1, 1, grid_size])

    for (i, hm) in enumerate(heatmaps):
        w_sum = cupy.sum(hm, axis=[1, 2, 3])

        avg_x = cupy.sum(x * hm, axis=[1, 2, 3]) / w_sum
        avg_y = cupy.sum(y * hm, axis=[1, 2, 3]) / w_sum
        avg_z = cupy.sum(z * hm, axis=[1, 2, 3]) / w_sum

        p3D = cupy.array([avg_x, avg_y, avg_z], dtype=numpy.float32).T
        ret[i] = p3D

    return ret


def heatMaps2Points3DGPU_max(heatmaps, Nvox=44, cube=(250, 250, 250)):
    # heatmaps have shape (batch, num_joints, Nvox, Nvox, Nvox)
    if len(heatmaps.shape) < 5:
        heatmaps = cupy.expand_dims(heatmaps, 0)

    ret = cupy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=cupy.float32)

    # go through batch data
    for (i, hm) in enumerate(heatmaps):
        # go through joints
        for (j, h) in enumerate(hm):
            ind = cupy.array(cupy.unravel_index(cupy.argmax(h, axis=None), h.shape), dtype=cupy.float32)
            ret[i, j, :] = (ind / float(Nvox) - 0.5) * cube[0]

    return ret


def heatMaps2VoxelGPU_max(heatmaps, Nvox=44):
    # heatmaps have shape (batch, num_joints, Nvox, Nvox, Nvox)
    if len(heatmaps.shape) < 5:
        heatmaps = cupy.expand_dims(heatmaps, 0)

    ret = cupy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=cupy.float32)

    # go through batch data
    for (i, hm) in enumerate(heatmaps):
        # go through joints
        for (j, h) in enumerate(hm):
            ind = cupy.array(cupy.unravel_index(cupy.argmax(h, axis=None), h.shape), dtype=cupy.float32)
            ret[i, j, :] = ind

    return ret


def heatMaps2Points3DGPU_smooth_max(heatmaps, Nvox=44, cube=(250, 250, 250)):
    # heatmaps have shape (batch, num_joints, Nvox, Nvox, Nvox)
    if len(heatmaps.shape) < 5:
        heatmaps = cupy.expand_dims(heatmaps, 0)

    ret = cupy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=cupy.float32)

    # go through batch data
    for (i, hm) in enumerate(heatmaps):
        # go through joints
        for (j, h) in enumerate(hm):
            ind = cupy.array(cupy.unravel_index(cupy.argmax(h, axis=None), h.shape), dtype=cupy.float32)
            # send the local submatrix to compute the weighted average
            H = cupy.zeros((3, 3, 3), dtype=cupy.float32)
            for a in range(-1, 2):
                for b in range(-1, 2):
                    for c in range(-1, 2):
                        if ind[0] + a >= 0 and ind[0] + a < Nvox and ind[1] + b >= 0 and ind[1] + b < Nvox and ind[
                            2] + c >= 0 and ind[2] + c < Nvox:
                            H[a + 1, b + 1, c + 1] = h[int(ind[0]) + a, int(ind[1]) + b, int(ind[2]) + c]
                        else:
                            H[a + 1, b + 1, c + 1] = 0

            H = cupy.expand_dims(H, 0)
            ind_s = heatMaps2VoxelGPU(H, Nvox=3)[0].T
            ind = cupy.expand_dims(ind, 1)
            A = ind + ind_s - 1
            ret[i, j, :] = (A[:, 0] / float(Nvox) - 0.5) * cube[0]

    return ret


def heatMaps2Points3D(heatmaps, cubes, Nvox=44):
    if len(heatmaps.shape) < 5:
        heatmaps = numpy.expand_dims(heatmaps, 0)

    ret = numpy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=numpy.float32)

    grid_size = Nvox
    x = numpy.arange(grid_size, dtype=cupy.float32).reshape([1, grid_size, 1, 1])
    y = numpy.arange(grid_size, dtype=cupy.float32).reshape([1, 1, grid_size, 1])
    z = numpy.arange(grid_size, dtype=cupy.float32).reshape([1, 1, 1, grid_size])

    for (i, hm) in enumerate(heatmaps):
        w_sum = numpy.sum(hm, axis=(1, 2, 3))
        avg_x = numpy.sum(x * hm, axis=(1, 2, 3)) / w_sum
        avg_y = numpy.sum(y * hm, axis=(1, 2, 3)) / w_sum
        avg_z = numpy.sum(z * hm, axis=(1, 2, 3)) / w_sum

        avg_x = (avg_x / float(Nvox) - 0.5) * cubes[i][0]
        avg_y = (avg_y / float(Nvox) - 0.5) * cubes[i][1]
        avg_z = (avg_z / float(Nvox) - 0.5) * cubes[i][2]

        p3D = numpy.array([avg_x, avg_y, avg_z], dtype=numpy.float32).T
        ret[i] = p3D

    return ret


def heatMaps2Voxel(heatmaps, Nvox=44):
    if len(heatmaps.shape) < 5:
        heatmaps = numpy.expand_dims(heatmaps, 0)

    ret = numpy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=numpy.float32)

    grid_size = Nvox
    x = numpy.arange(grid_size, dtype=cupy.float32).reshape([1, grid_size, 1, 1])
    y = numpy.arange(grid_size, dtype=cupy.float32).reshape([1, 1, grid_size, 1])
    z = numpy.arange(grid_size, dtype=cupy.float32).reshape([1, 1, 1, grid_size])

    for (i, hm) in enumerate(heatmaps):
        w_sum = numpy.sum(hm, axis=(1, 2, 3))
        avg_x = numpy.sum(x * hm, axis=(1, 2, 3)) / w_sum
        avg_y = numpy.sum(y * hm, axis=(1, 2, 3)) / w_sum
        avg_z = numpy.sum(z * hm, axis=(1, 2, 3)) / w_sum

        p3D = numpy.array([avg_x, avg_y, avg_z], dtype=numpy.float32).T
        ret[i] = p3D

    return ret


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


def heatMaps2Points3D_max(heatmaps, cubes, Nvox=44):
    # heatmaps have shape (batch, num_joints, Nvox, Nvox, Nvox)
    if len(heatmaps.shape) < 5:
        heatmaps = numpy.expand_dims(heatmaps, 0)

    ret = numpy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=numpy.float32)

    # go through batch data
    for (i, hm) in enumerate(heatmaps):
        # go through joints
        for (j, h) in enumerate(hm):
            ind = numpy.array(numpy.unravel_index(numpy.argmax(h, axis=None), h.shape))
            ret[i, j, :] = (ind / float(Nvox) - 0.5) * cubes[i][0]

    return ret


def heatMaps2Voxel_max(heatmaps, Nvox=44):
    # heatmaps have shape (batch, num_joints, Nvox, Nvox, Nvox)
    if len(heatmaps.shape) < 5:
        heatmaps = numpy.expand_dims(heatmaps, 0)

    ret = numpy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=numpy.float32)

    # go through batch data
    for (i, hm) in enumerate(heatmaps):
        # go through joints
        for (j, h) in enumerate(hm):
            ind = numpy.array(numpy.unravel_index(numpy.argmax(h, axis=None), h.shape))
            ret[i, j, :] = ind

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


def heatMaps2Points3D_smooth_max_var(heatmaps, cubes, Nvox=44, smooth_size=3):
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
            ind_s = heatMaps2Voxel(H, Nvox=smooth_size)[0]
            ret[i, j, :] = ((ind + ind_s - center) / float(Nvox) - 0.5) * cubes[i][0]

            ret_conf[i, j] = interpn((x1, x2, x3), H[0], ind_s, bounds_error=False, fill_value=None)

    return Variable(ret), ret_conf


def heatMaps2NbestPoints3D_smooth_max(heatmaps, cubes, Nvox=44, smooth_size=3):
    # heatmaps have shape (batch, num_joints, Nvox, Nvox, Nvox)
    if len(heatmaps.shape) < 5:
        heatmaps = numpy.expand_dims(heatmaps, 0)

    # smooth_size needs to be odd
    if smooth_size % 2 == 0:
        smooth_size += 1

    smooth_size2 = int(smooth_size // 2)

    ret = numpy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3), dtype=numpy.float32)
    ret_conf = numpy.zeros((heatmaps.shape[0], heatmaps.shape[1]), dtype=numpy.float32)

    smoot_axis = numpy.arange(-smooth_size2, smooth_size2 + 1)
    idxX, idxY, idxZ = numpy.meshgrid(smoot_axis, smoot_axis, smoot_axis)

    # go through batch data
    for (i, hm) in enumerate(heatmaps):
        # go through joints
        for (j, h) in enumerate(hm):
            # inds = []
            # hh = h.copy()
            # for k in range(Nbest):

            ind = numpy.array(numpy.unravel_index(numpy.argmax(h, axis=None), h.shape))
            # hh[ind[0],ind[1],ind[2]] = 0.
            # inds.append(ind)

            # uprava kdyz je predikce moc na kraji
            if ind[0] - smooth_size2 < 0:
                ind[0] = smooth_size2
            if ind[1] - smooth_size2 < 0:
                ind[1] = smooth_size2
            if ind[2] - smooth_size2 < 0:
                ind[2] = smooth_size
            if ind[0] + smooth_size2 + 1 > Nvox:
                ind[0] = Nvox - smooth_size2 - 1
            if ind[1] + smooth_size2 + 1 > Nvox:
                ind[1] = Nvox - smooth_size2 - 1
            if ind[2] + smooth_size2 + 1 > Nvox:
                ind[2] = Nvox - smooth_size2 - 1

            w = h[ind[0] - smooth_size2:ind[0] + smooth_size2 + 1,
                ind[1] - smooth_size2:ind[1] + smooth_size2 + 1,
                ind[2] - smooth_size2:ind[2] + smooth_size2 + 1]
            if numpy.sum(w) != 0.0:

                x = numpy.average(idxX + ind[0], weights=w)
                y = numpy.average(idxY + ind[1], weights=w)
                z = numpy.average(idxZ + ind[2], weights=w)

                ret[i, j, :] = (numpy.array([x, y, z]) / float(Nvox) - 0.5) * cubes[i][0]
                ret_conf[i, j] = numpy.average(w, weights=w)  # pozor, vysledek je prumer z w^2

            else:
                # dej do stredu
                ret[i, j, :] = numpy.zeros((1, 3), dtype=numpy.float32)
                ret_conf[i, j] = 0.0

    return ret, ret_conf


def heatMaps2NbestPoints3D_smooth_max2(heatmaps, cubes, Nvox=44, smooth_size=3, Nbest=1):
    # heatmaps have shape (batch, num_joints, Nvox, Nvox, Nvox)
    if len(heatmaps.shape) < 5:
        heatmaps = numpy.expand_dims(heatmaps, 0)

    # smooth_size needs to be odd
    if smooth_size % 2 == 0:
        smooth_size += 1

    smooth_size2 = int(smooth_size // 2)

    ret = numpy.zeros((heatmaps.shape[0], heatmaps.shape[1], 3, Nbest), dtype=numpy.float32)
    ret_conf = numpy.zeros((heatmaps.shape[0], heatmaps.shape[1], Nbest), dtype=numpy.float32)

    smoot_axis = numpy.arange(-smooth_size2, smooth_size2 + 1)
    idxX, idxY, idxZ = numpy.meshgrid(smoot_axis, smoot_axis, smoot_axis)

    # go through batch data
    for (i, hm) in enumerate(heatmaps):
        # go through joints
        for (j, h) in enumerate(hm):
            # for N best
            for k in range(Nbest):
                # ss = time.time()
                ind = numpy.array(numpy.unravel_index(numpy.argmax(h, axis=None), h.shape))
                # uprava kdyz je predikce moc na kraji
                if ind[0] - smooth_size2 < 0:
                    ind[0] = smooth_size2
                if ind[1] - smooth_size2 < 0:
                    ind[1] = smooth_size2
                if ind[2] - smooth_size2 < 0:
                    ind[2] = smooth_size2
                if ind[0] + smooth_size2 + 1 > Nvox:
                    ind[0] = Nvox - smooth_size2 - 1
                if ind[1] + smooth_size2 + 1 > Nvox:
                    ind[1] = Nvox - smooth_size2 - 1
                if ind[2] + smooth_size2 + 1 > Nvox:
                    ind[2] = Nvox - smooth_size2 - 1

                w = h[ind[0] - smooth_size2:ind[0] + smooth_size2 + 1,
                    ind[1] - smooth_size2:ind[1] + smooth_size2 + 1,
                    ind[2] - smooth_size2:ind[2] + smooth_size2 + 1]

                if numpy.sum(w) != 0.0:

                    x = numpy.average(idxX + ind[0], weights=w)
                    y = numpy.average(idxY + ind[1], weights=w)
                    z = numpy.average(idxZ + ind[2], weights=w)

                    ret[i, j, :, k] = (numpy.array([x, y, z]) / float(Nvox) - 0.5) * cubes[i][0]
                    ret_conf[i, j, k] = numpy.average(w, weights=w)  # pozor, vysledek je prumer z w^2

                else:
                    # dej do stredu
                    ret[i, j, :, k] = numpy.zeros((1, 3), dtype=numpy.float32)
                    ret_conf[i, j, k] = 0.0

                # vymaskovat nalezene maximum a muzes hledat dalsi max
                h[ind[0], ind[1], ind[2]] = 0.0
                # print(time.time() - ss)
            # sys.exit()

    return ret, ret_conf


# computes the mean error in mm for the whole hand
def meanErrorOnHand(points3D_ref, points3D_pred):
    err = numpy.mean(numpy.sqrt(numpy.sum(numpy.square(points3D_pred[:, :, :] - points3D_ref[:, :, :]), axis=2)))

    return err


def heatMapsMask(heatmapsmask, idx, which, device=0):
    # heatmaps have shape (batch, num_joints, Nvox, Nvox, Nvox)
    # with cupy.cuda.Device(device):
    ## go through batch data
    ##for (i, hm) in enumerate(heatmaps):
    ## go through joints
    # heatmapsmask = cupy.copy(heatmaps)
    # if whichmask == -1:
    #  whichmask = random.randint(0,4)
    with cupy.cuda.Device(device):
        # print('heatmapsmask data', heatmapsmask.device)
        # print('idc]x data', idx.device)
        heatmapsmask[:, idx[cupy.int32(which)], :, :, :] = cupy.float32(0.0)

    return heatmapsmask