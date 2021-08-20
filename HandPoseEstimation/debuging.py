import h5py
import io
from Utils import v2v_misc
import numpy as np
import mayavi.mlab


def visualisation3(V, labelStack, Nvox_data=88, Nvox_label=88):
    ##### visualisation #####

    ratio = Nvox_data / float(Nvox_label)
    yy, xx, zz = np.where(V == True)
    mayavi.mlab.points3d(xx, yy, zz,
                         mode="cube",
                         color=(0, 1, 0),
                         scale_factor=0.75)
    # mayavi.mlab.show()

    # zaokrouhlit na int
    labelStack_resized = (labelStack - Nvox_label / 2.0) * ratio + Nvox_data / 2.0
    indL = np.asarray(np.rint(labelStack_resized), dtype='int')
    indL = v2v_misc.range_assert(indL, Nvox_data)
    n = len(indL)
    U = np.zeros((Nvox_data, Nvox_data, Nvox_data), dtype='bool')
    for j in range(n):
        U[indL[j][0], indL[j][1], indL[j][2]] = True

    # vizualizace
    xx, yy, zz = np.where(U == True)
    mayavi.mlab.points3d(xx, yy, zz,
                         mode="cube",
                         color=(1, 0, 0),
                         scale_factor=1.2)
    mayavi.mlab.show()


def visualisation2(V, labelStack, heatMaps, Nvox_data=88, Nvox_label_heatMap=44):
    ##### visualisation #####

    ratio = Nvox_data / float(Nvox_label_heatMap)
    yy, xx, zz = np.where(V == True)
    mayavi.mlab.points3d(xx, yy, zz,
                         mode="cube",
                         color=(0, 1, 0),
                         scale_factor=0.75)
    # mayavi.mlab.show()

    # zaokrouhlit na int
    labelStack_resized = (labelStack - Nvox_label_heatMap / 2.0) * ratio + Nvox_data / 2.0
    indL = np.asarray(np.rint(labelStack_resized), dtype='int')
    indL = v2v_misc.range_assert(indL, Nvox_data)
    n = len(indL)
    U = np.zeros((Nvox_data, Nvox_data, Nvox_data), dtype='bool')
    for j in range(n):
        U[indL[j][2], indL[j][1], indL[j][0]] = True

        # vizualizace
    xx, yy, zz = np.where(U == True)
    mayavi.mlab.points3d(xx, yy, zz,
                         mode="cube",
                         color=(1, 0, 0),
                         scale_factor=1.2)
    # mayavi.mlab.show()

    ratio = Nvox_data / float(Nvox_label_heatMap)
    for i in range(heatMaps.shape[0]):
        # vizualizace
        E = heatMaps[i, :, :, :]
        xx, yy, zz = np.where(E > 0.2)
        xx = (xx - Nvox_label_heatMap / 2.0) * ratio + Nvox_data / 2.0
        yy = (yy - Nvox_label_heatMap / 2.0) * ratio + Nvox_data / 2.0
        zz = (zz - Nvox_label_heatMap / 2.0) * ratio + Nvox_data / 2.0
        mayavi.mlab.points3d(xx, yy, zz,
                             mode="cube",
                             color=(1, 1, 0),
                             scale_factor=0.5)
    mayavi.mlab.show()


#data_train = h5py.File(r'w:\cv\hpoes2\data\HANDS2019\train_com94_voxels.h5', 'r')
data_train = h5py.File(r'w:\cv\hpoes2\data\NYU\train_comrefV2V_voxels.h5', 'r')

key = 'real_voxels'

batch_data = []
batch_labels = []

for i in [0, 2]:
    pdata = data_train[key][str(i)][:].tostring()
    _file = io.BytesIO(pdata)
    data = np.load(_file)['arr_0']

    labels = data_train['labels'][i]

    batch_data.append(data)
    batch_labels.append(labels)

(batch_data_aug, batch_labels_aug) = v2v_misc.augmentation_volumetric(batch_data, batch_labels, repetitions=3, grid_size_label=88, app_thres=0.0, poly_order=0)
batch_labels_np = np.array(batch_labels_aug)
target_gpu = v2v_misc.make_global_heat_map_gpu(batch_labels_np, device=0)

for i in range(batch_data_aug.shape[0]):
    #visualisation3(batch_data_aug[i], batch_labels_aug[i], Nvox_label=88)
    visualisation2(batch_data_aug[i], batch_labels_aug[i], target_gpu[i], Nvox_label_heatMap=88)

