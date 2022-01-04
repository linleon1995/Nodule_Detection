'''
Modified Date: 2021/12/14
Author: Li-Wei Hsiao
mail: nfsmw308@gmail.com
'''
import copy
import csv
import functools
import glob
import math
import os
import cv2
import tqdm
import random
import matplotlib.pyplot as plt

from collections import namedtuple
from types import coroutine

import SimpleITK as sitk
import numpy as np
import scipy.ndimage.morphology as morph

import torch
import torch.cuda
import torch.nn.functional as F
from torch.utils.data import Dataset

from LUNA16_test.disk import getCache
# from disk import getCache
from LUNA16_test.util import XyzTuple, xyz2irc
# from util import XyzTuple, xyz2irc
from LUNA16_test.logconf import logging
# from logconf import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

raw_cache = getCache('part2segment')

MaskTuple = namedtuple('MaskTuple', 'raw_dense_mask, dense_mask, body_mask, air_mask, raw_candidate_mask, candidate_mask, lung_mask, neg_mask, pos_mask')

CandidateInfoTuple = namedtuple('CandidateInfoTuple', 'isNodule_bool, hasAnnotation_bool, diameter_mm, series_uid, center_xyz')

ROOT = rf'C:\Users\test\Desktop\Leon\Datasets'
data_subset_path = os.path.join(ROOT, 'LUNA16/data/subset*')
seg_lung_path = os.path.join(ROOT, 'LUNA16/data/seg-lungs-LUNA16')
annotations_path = os.path.join(ROOT, 'LUNA16/data/annotations.csv')
candidates_path = os.path.join(ROOT, 'LUNA16/data/candidates.csv')
#  +++
# TEST_SUBJECT = ['subset8', 'subset9']
TEST_SUBJECT = None
#  +++

@functools.lru_cache(1)
def getCandidateInfoList(requireOnDisk_bool=True):
    # We construct a set with all series_uids that are present on disk.
    # This will let us use the data, even if we haven't downloaded all of
    # the subsets yet.
    mhd_list = glob.glob(data_subset_path + '/*.mhd')
    #  +++
    if TEST_SUBJECT:
        new_mhd_list = []
        for path in mhd_list:
            for test_subject in TEST_SUBJECT:
                if test_subject in path:
                    new_mhd_list.append(path)
                    break
        mhd_list = new_mhd_list
    #  +++

    presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in mhd_list}

    lung_mhd_list = glob.glob(seg_lung_path + '/*.mhd')
    #  +++
    if TEST_SUBJECT:
        new_mhd_list = []
        for path in lung_mhd_list:
            for test_subject in TEST_SUBJECT:
                if test_subject in path:
                    new_mhd_list.append(path)
                    break
        lung_mhd_list = new_mhd_list
    #  +++
    lung_presentOnDisk_set = {os.path.split(p)[-1][:-4] for p in lung_mhd_list}

    candidateInfo_list = []
    with open(annotations_path, "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in presentOnDisk_set and series_uid not in lung_presentOnDisk_set and requireOnDisk_bool:
                continue
            
            annotationCenter_xyz = tuple([float(x) for x in row[1:4]])
            annotationDiameter_mm = float(row[4])

            candidateInfo_list.append(
                CandidateInfoTuple(
                    True,
                    True,
                    annotationDiameter_mm,
                    series_uid,
                    annotationCenter_xyz,
                )
            )

    with open(candidates_path, "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]

            if series_uid not in presentOnDisk_set and requireOnDisk_bool:
                continue

            isNodule_bool = bool(int(row[4]))
            candidateCenter_xyz = tuple([float(x) for x in row[1:4]])

            if not isNodule_bool:
                candidateInfo_list.append(
                    CandidateInfoTuple(
                        False,
                        False,
                        0.0,
                        series_uid,
                        candidateCenter_xyz,
                    )
                )

    candidateInfo_list.sort(reverse=True)
    return candidateInfo_list


@functools.lru_cache(1)
def getCandidateInfoDict(requireOnDisk_bool=True):
    candidateInfo_list = getCandidateInfoList(requireOnDisk_bool)
    candidateInfo_dict = {}

    for candidateInfo_tup in candidateInfo_list:
        candidateInfo_dict.setdefault(candidateInfo_tup.series_uid,
                                      []).append(candidateInfo_tup)

    return candidateInfo_dict


class Ct:
    def __init__(self, series_uid):
        mhd_path = glob.glob('{}/{}.mhd'.format(data_subset_path, series_uid))[0]

        ct_mhd = sitk.ReadImage(mhd_path)
        self.hu_a = np.array(sitk.GetArrayFromImage(ct_mhd), dtype=np.float32)

        lung_mhd_path = glob.glob('{}/{}.mhd'.format(seg_lung_path, series_uid))[0]

        lung_ct_mhd = sitk.ReadImage(lung_mhd_path)
        self.lung_mask = np.array(sitk.GetArrayFromImage(lung_ct_mhd), dtype=np.float32)

        self.series_uid = series_uid

        self.origin_xyz = XyzTuple(*ct_mhd.GetOrigin())
        self.vxSize_xyz = XyzTuple(*ct_mhd.GetSpacing())
        self.direction_a = np.array(ct_mhd.GetDirection()).reshape(3, 3)

        candidateInfo_list = getCandidateInfoDict()[self.series_uid]

        self.positiveInfo_list = [
            candidate_tup
            for candidate_tup in candidateInfo_list
            if candidate_tup.isNodule_bool
        ]
        self.positive_mask = self.buildAnnotationMask(self.positiveInfo_list)
        self.positive_indexes = (self.positive_mask.sum(axis=(1,2))
                                 .nonzero()[0].tolist())

    def buildAnnotationMask(self, positiveInfo_list, threshold_hu = -700):
        boundingBox_a = np.zeros_like(self.hu_a, dtype=np.bool)

        for candidateInfo_tup in positiveInfo_list:
            center_irc = xyz2irc(
                candidateInfo_tup.center_xyz,
                self.origin_xyz,
                self.vxSize_xyz,
                self.direction_a,
            )
            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.col)

            index_radius = 2
            try:
                while self.hu_a[ci + index_radius, cr, cc] > threshold_hu and \
                        self.hu_a[ci - index_radius, cr, cc] > threshold_hu:
                    index_radius += 1
            except IndexError:
                index_radius -= 1

            row_radius = 2
            try:
                while self.hu_a[ci, cr + row_radius, cc] > threshold_hu and \
                        self.hu_a[ci, cr - row_radius, cc] > threshold_hu:
                    row_radius += 1
            except IndexError:
                row_radius -= 1

            col_radius = 2
            try:
                while self.hu_a[ci, cr, cc + col_radius] > threshold_hu and \
                        self.hu_a[ci, cr, cc - col_radius] > threshold_hu:
                    col_radius += 1
            except IndexError:
                col_radius -= 1

            boundingBox_a[
                 ci - index_radius: ci + index_radius + 1,
                 cr - row_radius: cr + row_radius + 1,
                 cc - col_radius: cc + col_radius + 1] = True
        self.lung_mask[self.lung_mask == 3] = 1
        self.lung_mask[self.lung_mask == 4] = 1
        self.lung_mask[self.lung_mask != 1] = 0
        mask_a = boundingBox_a & (self.hu_a > threshold_hu) & self.lung_mask.astype('bool')

        return mask_a

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


@functools.lru_cache(1, typed=True)
def getCt(series_uid):
    return Ct(series_uid)

@raw_cache.memoize(typed=True)
def getCtRawCandidate(series_uid, center_xyz, width_irc):
    ct = getCt(series_uid)
    ct_chunk, pos_chunk, center_irc = ct.getRawCandidate(center_xyz,
                                                         width_irc)
    ct_chunk.clip(-1000, 1000, ct_chunk)
    return ct_chunk, pos_chunk, center_irc

@raw_cache.memoize(typed=True)
def getCtSampleSize(series_uid):
    ct = Ct(series_uid)
    return int(ct.hu_a.shape[0]), ct.positive_indexes


class Luna2dSegmentationDataset(Dataset):
    def __init__(self,
                 val_stride=0,
                 isValSet_bool=None,
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
            self.series_list = sorted(getCandidateInfoDict().keys())

        if isValSet_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        self.sample_list = []
        for series_uid in tqdm.tqdm( self.series_list ):
            if glob.glob('{}/{}.mhd'.format(data_subset_path, series_uid)) == []:
                continue
            index_count, positive_indexes = getCtSampleSize(series_uid)
            if self.fullCt_bool:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in range(index_count) if slice_ndx%self.contextSlices_shift == 0 or self.contextSlices_shift == 0]
            else:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in positive_indexes if slice_ndx%self.contextSlices_shift == 0 or self.contextSlices_shift == 0]

        self.candidateInfo_list = getCandidateInfoList()

        series_set = set(self.series_list)
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list
                                   if cit.series_uid in series_set]

        self.pos_list = [nt for nt in self.candidateInfo_list
                            if nt.isNodule_bool]

        log.info("{!r}: {} {} series, {} slices, {} nodules".format(
            self,
            len(self.series_list),
            {None: 'general', True: 'validation', False: 'training'}[isValSet_bool],
            len(self.sample_list),
            len(self.pos_list),
        ))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx]
        return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid)
        
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
        return ct_t, pos_t, ct.series_uid, slice_ndx


class TrainingLuna2dSegmentationDataset(Luna2dSegmentationDataset):
    def __init__(self, n_class = 2, shift = 64, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 2
        self.shift = shift
        self.n_class = n_class

    def __len__(self):
        return len(self.sample_list)

    def shuffleSamples(self):
        random.shuffle(self.sample_list)
        

    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx]
        return self.getitem_trainingCrop(series_uid, slice_ndx)

    def getitem_trainingCrop(self, series_uid, slice_ndx):
        
        ct = getCt(series_uid)
        
        ct_a = torch.zeros((self.layers, self.img_size, self.img_size))

        start_ndx = slice_ndx - self.contextSlices_count if slice_ndx - self.contextSlices_count >= 0 else 0
        start_ndx = ct.positive_mask.shape[0] - self.layers if start_ndx + self.layers >= ct.positive_mask.shape[0] else start_ndx
        
        end_ndx = slice_ndx + self.contextSlices_count + 1 if slice_ndx + self.contextSlices_count + 1 <= ct.positive_mask.shape[0] else ct.positive_mask.shape[0]
        end_ndx = self.layers if end_ndx < self.layers else end_ndx

        # print(f'{series_uid}  slice:{slice_ndx}   ndx:{ct.positive_mask.shape[0]}   range:{start_ndx}:{end_ndx}')
        
        for i, context_ndx in enumerate(range(start_ndx, end_ndx)):
            context_ndx = max(context_ndx, 0)
            context_ndx = min(context_ndx, ct.hu_a.shape[0] - 1)
            ct_a[i] = torch.from_numpy(ct.hu_a[context_ndx].astype(np.float32))

        ct_a.clamp_(-1000, 1000)


        pos_a = torch.from_numpy(ct.positive_mask[slice_ndx])

        ct_t = torch.zeros((self.layers, self.img_size + self.shift, self.img_size + self.shift))
        pos_t = torch.zeros((self.img_size + self.shift, self.img_size + self.shift))
        ct_t[:,:,:] = -1000
        ct_t[:, int(self.shift/2):int(self.shift/2) + self.img_size, int(self.shift/2):int(self.shift/2) + self.img_size] = ct_a
        pos_t[int(self.shift/2):int(self.shift/2) + self.img_size, int(self.shift/2):int(self.shift/2) + self.img_size] = pos_a
        row_offset = random.randrange(0,self.shift)
        col_offset = random.randrange(0,self.shift)
        ct_t = ct_t[:, row_offset:row_offset + self.img_size, col_offset:col_offset + self.img_size]
        pos_t = pos_t[row_offset:row_offset + self.img_size, col_offset:col_offset + self.img_size]

        for i in range(0, self.n_class):
            gt = np.zeros((pos_t.shape[0], pos_t.shape[1]))
            gt[pos_t[:, :] == i] = 1
            if i == 0:
                gts = gt[np.newaxis, :, :]
            else:
                gts = np.concatenate((gts, gt[np.newaxis, :, :]), axis = 0)
        pos_ts = torch.as_tensor(gts.astype('int'))

        return ct_t, pos_t, pos_ts, ct.series_uid, slice_ndx


class TestingLuna2dSegmentationDataset(Dataset):
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
            self.series_list = sorted(getCandidateInfoDict().keys())

        self.sample_list = []
        for series_uid in tqdm.tqdm( self.series_list ):
            if glob.glob('{}/{}.mhd'.format(data_subset_path, series_uid)) == []:
                continue
            index_count, positive_indexes = getCtSampleSize(series_uid)

            if self.fullCt_bool:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in range(index_count) if slice_ndx%self.contextSlices_shift == 0]
            else:
                self.sample_list += [(series_uid, slice_ndx) for slice_ndx in positive_indexes if slice_ndx%self.contextSlices_shift == 0]

        self.candidateInfo_list = getCandidateInfoList()

        series_set = set(self.series_list)
        self.candidateInfo_list = [cit for cit in self.candidateInfo_list
                                   if cit.series_uid in series_set]

        self.pos_list = [nt for nt in self.candidateInfo_list
                            if nt.isNodule_bool]

        log.info("[Test]  {} series, {} slices, {} nodules".format(
            len(self.series_list),
            len(self.sample_list),
            len(self.pos_list),
        ))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, ndx):
        series_uid, slice_ndx = self.sample_list[ndx]
        return self.getitem_fullSlice(series_uid, slice_ndx)

    def getitem_fullSlice(self, series_uid, slice_ndx):
        ct = getCt(series_uid)
        
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
        return ct_t, pos_t, ct.series_uid, slice_ndx

    