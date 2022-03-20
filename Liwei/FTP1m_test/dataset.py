import os, cv2, tqdm, random
import SimpleITK as sitk
import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset
from Liwei.FTP1m_test.util import irc2xyz, xyz2irc, XyzTuple

data_path = './malignant.csv'

class Ct:
    def __init__(self, series_uid, raw_path, mask_path):
        ct_mhd = sitk.ReadImage(raw_path)
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)
        self.series_uid = series_uid

        mask_mhd = sitk.ReadImage(mask_path)
        self.positive_mask = np.array(sitk.GetArrayFromImage(mask_mhd), dtype=np.float32)
        self.origin_xyz = XyzTuple(*mask_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*mask_mhd.GetSpacing())
        self.direction_a = np.array(mask_mhd.GetDirection()).reshape(3, 3)

        self.positive_indexes = (self.positive_mask.sum(axis=(1,2)).nonzero()[0].tolist())

        # if show:
        #     for i in range(self.hu_a.shape[0]):
        #         plt.imshow(self.hu_a[i])
        #         contours, hierarchy = cv2.findContours(self.positive_mask[i].astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        #         draw = np.zeros((self.positive_mask[i].shape[0], self.positive_mask[i].shape[1], 3))
        #         for idx, contour in enumerate( contours ):
        #             if contour.ndim != 1:
        #                 cv2.drawContours(draw, contours, idx, (0,0,255), 1)
        #         plt.imshow(draw.astype('uint8'), alpha=0.3)
        #         plt.axis('off')
        #         plt.show()

    def getRawCandidate(self, center_xyz, width_irc):
        center_irc = xyz2irc(center_xyz, self.origin_xyz, self.vxSize_xyz,
                             self.direction_a)

        slice_list = []
        for axis, center_val in enumerate(center_irc):
            start_ndx = int(round(center_val - width_irc[axis]/2))
            end_ndx = int(start_ndx + width_irc[axis])

            assert center_val >= 0 and center_val < self.hu_a.shape[axis], repr([self.series_uid, center_xyz, self.origin_xyz, self.vxSize_xyz, center_irc, axis])

            if start_ndx < 0:
                start_ndx = 0
                end_ndx = int(width_irc[axis])

            if end_ndx > self.hu_a.shape[axis]:
                end_ndx = self.hu_a.shape[axis]
                start_ndx = int(self.hu_a.shape[axis] - width_irc[axis])

            slice_list.append(slice(start_ndx, end_ndx))

        ct_chunk = self.hu_a[tuple(slice_list)]
        pos_chunk = self.positive_mask[tuple(slice_list)]

        return ct_chunk, pos_chunk, center_irc

def getCt(series_uid, raw_path, mask_path):
    return Ct(series_uid, raw_path, mask_path)

def getCtRawCandidate(series_uid, raw_path, mask_path, center_slice, width_irc):
    ct = getCt(series_uid, raw_path, mask_path)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_slice, width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc

def getCtSampleSize(series_uid, raw_path, mask_path):
    ct = Ct(series_uid, raw_path, mask_path)
    return int(ct.hu_a.shape[0]), ct.positive_indexes

def generate_seriesuid(data_path):
    from modules.data import dataset_utils
    raw_dirs = dataset_utils.get_files(data_path, 'raw', get_dirs=True)
    mask_dirs = dataset_utils.get_files(data_path, 'mask', get_dirs=True)

    for raw_dir, mask_dir in zip(raw_dirs, mask_dirs):
        raw_file = dataset_utils.get_files(raw_dir, 'mhd')[0]
        mask_file = dataset_utils.get_files(mask_dir, 'mhd')[0]
    return 

class FTP2dSegmentationDataset(Dataset):
    def __init__(self,
                 series_uid=None,
                 contextSlices_count = 3,
                 contextSlices_shift = 7,
                 fullCt_bool=False,
                 img_size = 512,
            ):
        self.contextSlices_count = contextSlices_count
        self.contextSlices_shift = contextSlices_shift
        self.layers = self.contextSlices_count * 2 + 1
        self.fullCt_bool = fullCt_bool
        self.img_size = img_size

        if series_uid:
            self.series_list = [series_uid]
        else:
            # self.series_list = pd.read_csv(data_path).values
            self.series_list = generate_seriesuid()

        self.sample_list = []
        self.nodule_center_slice = []
        self.nodule_center = []
        self.uid = set({})
        nodule = 0
        for case, series_uid, raw_path, mask_path, coodX, coodY, coodZ in tqdm.tqdm( self.series_list ):
            index_count, positive_indexes = getCtSampleSize(series_uid, raw_path, mask_path)
            nodule += len(positive_indexes)
            idx = positive_indexes[int(len(positive_indexes)/2)] if len(positive_indexes) != 1 else positive_indexes[0]
            self.nodule_center += [(case, series_uid, raw_path, mask_path, coodX, coodY, coodZ)]
            # print(positive_indexes, idx)
            if f'{case}_{series_uid}' not in self.uid:
                self.uid.add(f'{case}_{series_uid}')
                nodule += len(positive_indexes)
                idx = positive_indexes[int(len(positive_indexes)/2)] if len(positive_indexes) != 1 else positive_indexes[0]
                self.nodule_center_slice += [(case, series_uid, raw_path, mask_path)]
                if self.fullCt_bool:
                    self.sample_list += [(case, series_uid, raw_path, mask_path, slice_ndx) for slice_ndx in range(index_count) if slice_ndx%self.contextSlices_shift == 0]
                else:
                    self.sample_list += [(case, series_uid, raw_path, mask_path, slice_ndx) for slice_ndx in positive_indexes if slice_ndx%self.contextSlices_shift == 0]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        case, series_uid, raw_path, mask_path, slice_ndx = self.sample_list[ndx]
        return self.getitem_fullSlice(case, series_uid, raw_path, mask_path, slice_ndx)

    def getitem_fullSlice(self, case, series_uid, raw_path, mask_path, slice_ndx):
        ct = getCt(series_uid, raw_path, mask_path)
        
        ct_t = torch.zeros((self.layers, self.img_size, self.img_size))

        start_ndx = slice_ndx - self.contextSlices_count if slice_ndx - self.contextSlices_count >= 0 else 0
        start_ndx = ct.positive_mask.shape[0] - self.layers if start_ndx + self.layers >= ct.positive_mask.shape[0] else start_ndx
        
        end_ndx = slice_ndx + self.contextSlices_count + 1 if slice_ndx + self.contextSlices_count + 1 <= ct.positive_mask.shape[0] else ct.positive_mask.shape[0]
        end_ndx = self.layers if end_ndx < self.layers else end_ndx

        # print(f'{series_uid}  slice:{slice_ndx}   ndx:{ct.positive_mask.shape[0]}   range:{start_ndx}:{end_ndx}')
        
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        ct_t.clamp_(-1000, 1000)

        pos_t = torch.from_numpy(ct.positive_mask[slice_ndx])
        return ct_t, pos_t, ct.series_uid, slice_ndx, case

class TrainingFTP2dSegmentationDataset(FTP2dSegmentationDataset):
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.center_slice_list = self.sample_list if self.fullCt_bool else self.nodule_center
    
    def __len__(self):
        return len(self.center_slice_list)

    def shuffleSamples(self):
        random.shuffle(self.center_slice_list)

    def __getitem__(self, ndx):
        case, series_uid, raw_path, mask_path, coodX, coodY, coodZ = self.center_slice_list[ndx]
        return self.getitem_trainingCrop(case, series_uid, raw_path, mask_path, coodX, coodY, coodZ)

    def getitem_trainingCrop(self, case, series_uid, raw_path, mask_path, coodX, coodY, coodZ):
        ct_a, pos_a, center_irc = getCtRawCandidate(
            series_uid,
            raw_path,
            mask_path,
            (coodX, coodY, coodZ),
            (self.layers, self.img_size, self.img_size),
        )
        slice_ndx = center_irc.index

        if self.layers != 1:
            pos_a = pos_a[3:4]
        ct_t = torch.from_numpy(ct_a).to(torch.float32)
        pos_t = torch.from_numpy(pos_a).to(torch.long)

        return ct_t, pos_t, series_uid, slice_ndx, case


if __name__ in "__main__":
    contextSlices_count = 0
    save_path = './1m_nodule'
    os.makedirs(save_path, exist_ok=True)

    # ds = FTP2dSegmentationDataset(
    #         series_uid=('1m0043','1.2.826.0.1.3680043.2.1125.1.66267488139869463859646041266078917','../../dataset/FTPuser/malignant/1m0043/1m0043raw mhd/1.2.826.0.1.3680043.2.1125.1.66267488139869463859646041266078917.mhd','../../dataset/FTPuser/malignant/1m0043/1m0043mask mhd/1.2.826.0.1.3680043.2.1125.1.20492007384673651600845318549231386.mhd'),
    #         contextSlices_count=contextSlices_count,
    #         contextSlices_shift=1,
    #         fullCt_bool=False,
    #         img_size = 512
    #     )
    
    ds = TrainingFTP2dSegmentationDataset(
            series_uid=None,
            contextSlices_count=contextSlices_count,
            contextSlices_shift=1,
            fullCt_bool=False,
            img_size = 96
        )
    for idx, (ct_t, pos_t, series_uid, ct_ndx, case) in enumerate(ds):
        fig, axs = plt.subplots(1, contextSlices_count*2 + 1, figsize=(10, 10))
        if contextSlices_count > 0:
            for i in range(0, ct_t.shape[0]):
                axs[i].imshow(ct_t[i], cmap='gray')
                pos_t = pos_t.squeeze()
                if pos_t.sum() != 0:
                    axs[i].contour(pos_t, 10, cmap='Reds')
                axs[i].axis('off')
        else:
            pos_t = pos_t.squeeze()
            plt.imshow(ct_t[0], cmap='gray')
            if pos_t.sum() != 0:
                plt.contour(pos_t, 5, cmap='Greens')
            plt.axis('off')
        plt.savefig(f"{save_path}/{case}_{series_uid}_{ct_ndx}_{idx}.png")
        # plt.show()
        plt.close('all')
        if idx == len(ds):
            break

