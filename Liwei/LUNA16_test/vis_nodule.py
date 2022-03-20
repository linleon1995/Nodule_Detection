'''
Modified Date: 2021/12/14
Author: Li-Wei Hsiao
mail: nfsmw308@gmail.com
'''
import matplotlib
import os
import matplotlib.pyplot as plt
import numpy as np
from dataset_seg import Luna2dSegmentationDataset, TrainingLuna2dSegmentationDataset

if __name__ in "__main__":
    save_path = 'gt_plot'
    os.makedirs(save_path, exist_ok=True)
    contextSlices_count = 3
    # ds = TrainingLuna2dSegmentationDataset(
    #         val_stride=10,
    #         isValSet_bool=False,
    #         contextSlices_count=contextSlices_count,
    #         contextSlices_shift=4,
    #         fullCt_bool=True,
    #         img_size = 512,
    #         shift = 64
    #     ) 
    ds = Luna2dSegmentationDataset(
            val_stride=10,
            isValSet_bool=True,
            contextSlices_count=contextSlices_count,
            contextSlices_shift=1,
            fullCt_bool=False,
            img_size = 512,
        )
    for idx, (ct_t, pos_t, series_uid, ct_ndx) in enumerate(ds):
        fig, axs = plt.subplots(1, contextSlices_count*2 + 1, figsize=(10, 10))
        if contextSlices_count != 0:
            for i in range(0, ct_t.shape[0]):
                axs[i].imshow(ct_t[i], cmap='gray')
                if pos_t.sum() != 0:
                    axs[i].contour(pos_t, 10, cmap='Reds')
                axs[i].axis('off')
        else:
            plt.imshow(ct_t[0], cmap='gray')
            if pos_t.sum() != 0:
                plt.contour(pos_t, 10, cmap='Reds')
            plt.axis('off')
        plt.savefig(f"{save_path}/{series_uid}_{ct_ndx}.png")
        # plt.show()
        plt.close('all')
        if idx == len(ds):
            break
