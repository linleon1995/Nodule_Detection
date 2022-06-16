import numpy as np

from postprocessing.data_postprocess import VolumePostProcessor
from utils.nodule import LungNoduleStudy
from postprocessing.reduce_false_positive import NoduleClassifier


def get_nodule_id(pred_vol_category):
    return [pred_nodule_id for pred_nodule_id in np.unique(pred_vol_category)[1:]]


def remove_unusual_nodule_by_lung_size(pred_study, lung_mask_vol, min_lung_ration=0.5):
    lung_mask_pxiel_sum = np.sum(lung_mask_vol, axis=(1,2))
    ratio = lung_mask_pxiel_sum / np.max(lung_mask_pxiel_sum)
    remove_mask = np.where(ratio>=min_lung_ration, 0, 1)
    remove_mask = np.reshape(remove_mask, [remove_mask.size, 1, 1])

    pred_vol_category = pred_study.category_volume * remove_mask

    remove_nodule_ids = get_nodule_id(pred_vol_category)
    pred_study.record_nodule_removal(name='RUNLS', nodules_ids=remove_nodule_ids)
    return pred_study


def under_slice_removal(pred_study, slice_threshold=1):
    remove_nodule_ids = []
    for nodule_id, nodule in pred_study.nodule_instances.items():
        max_z = nodule.nodule_range['index']['max']
        min_z = nodule.nodule_range['index']['min']
        if max_z-min_z < slice_threshold:
            remove_nodule_ids.append(nodule_id)

    pred_study.record_nodule_removal(name='_1SR', nodules_ids=remove_nodule_ids)
    return pred_study


def simple_post_processor(vol, 
                          mask_vol, 
                          pred_vol,
                          lung_mask_vol,
                          pid,
                          FP_reducer_checkpoint,
                          _1SR,
                          RUNLS,
                          nodule_cls,
                          raw_vol=None,
                          connectivity=26,
                          area_threshold=8,
                          lung_size_threshold=0.4,
                          nodule_cls_prob=0.5,
                          crop_range=(32,64,64)
                          ):

    if raw_vol is None:
        raw_vol = vol

    # Data post-processing
    post_processer = VolumePostProcessor(connectivity, area_threshold)
    pred_vol_category = post_processer(pred_vol)
    # target_vol_category = post_processer(mask_vol)
    num_pred_nodule = np.unique(pred_vol_category).size-1
    # target_study = LungNoduleStudy(pid, target_vol_category, raw_volume=raw_vol)
    pred_study = LungNoduleStudy(pid, pred_vol_category, raw_volume=raw_vol)
    print(f'Predict Nodules (raw) {num_pred_nodule}')

    # False positive reducing
    if _1SR:
        pred_study = under_slice_removal(pred_study)

    if RUNLS:
        pred_study = remove_unusual_nodule_by_lung_size(pred_study, lung_mask_vol, min_lung_ration=lung_size_threshold)

    # Nodule classification
    if nodule_cls:
        crop_range = {'index': crop_range[0], 'row': crop_range[1], 'column': crop_range[2]}
        nodule_classifier = NoduleClassifier(
            crop_range, FP_reducer_checkpoint, prob_threshold=nodule_cls_prob)
        pred_study, pred_nodule_info = nodule_classifier.nodule_classify(
            vol, pred_study, mask_vol)
    return pred_study.category_volume
    