import numpy as np
import matplotlib.pyplot as plt
import cv2
import cc3d
import os


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


def build_nodule_distribution(ax, x, y, s, color, label):
    sc = ax.scatter(x, y, s=s, alpha=0.5, color=color, label=label)
    ax.set_title('The size and space distribution of lung nodules')
    ax.set_xlabel('row')
    ax.set_ylabel('column')
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.legend()
    # ax.legend(*sc.legend_elements("sizes", num=4))
    return ax


def single_nodule_distribution(ax, volume_list, color, label):
    size_list = []
    x, y, = [], []
    for volume in volume_list:
        total_nodule_info = build_nodule_metadata(volume)
        print(total_nodule_info)

        for nodule_info in total_nodule_info:
            size_list.append(nodule_info['Nodule_size'])
            x.append(np.int32(nodule_info['Noudle_center']['column']))
            y.append(np.int32(nodule_info['Noudle_center']['row']))
    ax = build_nodule_distribution(ax, x, y, size_list, color, label)
    return ax


def multi_nodule_distribution(train_volumes, test_volumes):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    path = rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\benign\raw\Image\1B004\1B004_0169.png'
    img = cv2.imread(path)
    ax.imshow(img)

    ax = single_nodule_distribution(ax, train_volumes, color='b', label='train')
    ax = single_nodule_distribution(ax, test_volumes, color='orange', label='test')

    fig.show()
    fig.savefig('lung.png')


def tmh_base_check(volume_generator, save_path=None):
    fig, ax = plt.subplots(1,1)
    total_size = []
    for vol_idx, (raw_vol, vol, mask_vol, infos) in enumerate(volume_generator):
        cat_vol = cc3d.connected_components(mask_vol, connectivity=26) # category volume
        nodule_ids = np.unique(cat_vol)[1:]
        pid = infos['pid']

        for n_id in nodule_ids:
            nodule_vol = np.int32(cat_vol==n_id)
            nodule_size = np.sum(nodule_vol)
            nodule_id = f'{pid}_{n_id:03d}'
            print(f'Nodule {nodule_id}  size {nodule_size} pixels')
            total_size.append({nodule_id: nodule_size})

            # Nodule HU changing
            if save_path is not None:
                zs, ys, xs = np.where(nodule_vol)
                unique_zs = np.unique(zs)
                # print(unique_zs, np.unique(ys), np.unique(xs))
                nodule_hu = []
                for z in unique_zs:
                    hu = np.mean(nodule_vol[z]*raw_vol[z,...,0])
                    nodule_hu.append(hu)
                ax.plot(nodule_hu)
                fig.savefig(os.path.join(save_path, f'{pid}_{n_id:03d}.png'))
                plt.cla()
                plt.clf()

    print(20*'-')
    nodule_sizes = list(total_size.values())
    max_size_nodule = list(total_size.keys())[list(total_size.values()).index(max(nodule_sizes))]
    print(f'Max size: {max_size_nodule} {max(nodule_sizes)}')
