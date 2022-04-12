import keras
import numpy as np
import matplotlib.pyplot as plt
from utils.evaluator import NoudleSegEvaluator
from data.volume_to_3d_crop import CropVolume
from data.crop_utils import crops_to_volume


class Keras3dSegEvaluator(NoudleSegEvaluator):
    def __init__(self, predictor, volume_generator, save_path, data_converter, eval_metrics, save_vis_condition=True, 
                 max_test_cases=None, post_processer=None, fp_reducer=None, nodule_classifier=None, lung_mask_path='./', 
                 save_all_images=False, batch_size=1, *args, **kwargs):
        super().__init__(predictor, volume_generator, save_path, data_converter, eval_metrics, save_vis_condition, 
                         max_test_cases, post_processer, fp_reducer, nodule_classifier, lung_mask_path, save_all_images, batch_size)
    
    def model_inference(self, vol, mask_vol):
        # TODO:
        crop_range, crop_shift, convert_dtype, overlapping = [64,64,32], [0,0,0], np.float32, 1.0
        cropping_op = self.data_converter(crop_range, crop_shift, convert_dtype, overlapping)

        vol = vol[...,0] / 255
        crop_infos = cropping_op(vol)
        crop_data_list = [info['data'][np.newaxis,np.newaxis] for info in crop_infos]

        crop_slice_list = []
        for info in crop_infos:
            slice_list = []
            for slice_range in info['slice']:
                # slice_list.append(slice(slice_range))
                slice_list.append(np.arange(*slice_range))
            crop_slice_list.append(slice_list)

        x_crops = np.concatenate(crop_data_list, axis=0)

        pred_crops = self.predictor.predict(x_crops)
        # for i, p in enumerate(x_crops):
        #     # if np.sum(p)>0.5:
        #     print(i, np.sum(p))
        #     plt.imshow(p[0,...,16])
        #     plt.show()
        pred_vol = crops_to_volume(pred_crops, crop_slice_list, vol.shape)

        class_vol = np.where(pred_vol>0.5, 1, 0)
        return class_vol