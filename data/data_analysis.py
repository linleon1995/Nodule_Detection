import numpy as np
import matplotlib.pyplot as plt
import cv2


def build_nodule_metadata(volume):
    if np.sum(volume) == np.sum(np.zeros_like(volume)):
        return None

    nodule_category = np.unique(volume)
    nodule_category = np.delete(nodule_category, np.where(nodule_category==0))
    total_nodule_metadata = []
    for label in nodule_category:
        binary_mask = volume==label
        nodule_size = np.sum(binary_mask)
        zs, ys, xs = np.where(binary_mask)
        center = {'index': np.mean(zs), 'row': np.mean(ys), 'column': np.mean(xs)}
        nodule_metadata = {'Nodule_id': label,
                            'Nodule_size': nodule_size,
                            'Nodule_slice': (np.min(zs), np.max(zs)),
                            'Noudle_center': center}
        total_nodule_metadata.append(nodule_metadata)
    return total_nodule_metadata


def build_nodule_dist(ax, x, y, s, color, label):
    sc = ax.scatter(x,y,s=s, alpha=0.5, label=label)
    ax.set_title('The size and space distribution of lung nodules')
    ax.set_xlabel('row')
    ax.set_ylabel('column')
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.legend()
    # ax.legend(*sc.legend_elements("sizes", num=4))
    return ax


def single_nodule_dist(ax, volume_list, color, label):
    size_list = []
    x, y, = [], []
    for volume in volume_list:
        total_nodule_info = build_nodule_metadata(volume)
        print(total_nodule_info)

        for nodule_info in total_nodule_info:
            size_list.append(nodule_info['Nodule_size'])
            x.append(np.int32(nodule_info['Noudle_center']['column']))
            y.append(np.int32(nodule_info['Noudle_center']['row']))
    ax = build_nodule_dist(ax, x, y, size_list, color, label)
    return ax


def multi_nodule_dist(train_volumes, test_volumes):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\benign\raw\Image\1B004\1B004_0169.png'
    img = cv2.imread(path)
    ax.imshow(img)

    ax = single_nodule_dist(ax, train_volumes, color='b', label='train')
    ax = single_nodule_dist(ax, test_volumes, color='orange', label='test')

    fig.show()
    fig.savefig('lung.png')
