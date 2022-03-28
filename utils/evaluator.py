import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data.data_structure import LungNoduleStudy
from utils.vis import save_mask, visualize, save_mask_in_3d, plot_scatter, ScatterVisualizer

from torch.utils.data import Dataset, DataLoader
    
class NoudleSegEvaluator():
    def __init__(self, predictor, volume_generator, save_path, data_converter, eval_metrics, save_vis_condition=True, 
                 max_test_cases=None, post_processer=None, fp_reducer=None, nodule_classifier=None, lung_mask_path='./', 
                 save_all_images=False, batch_size=1):
        self.predictor = predictor
        self.volume_generator = volume_generator
        self.save_path = save_path
        self.data_converter = data_converter
        self.eval_metrics = eval_metrics
        self.save_vis_condition = save_vis_condition
        self.max_test_cases = max_test_cases
        self.post_processer = post_processer
        self.fp_reducer = fp_reducer
        self.nodule_classifier = nodule_classifier
        self.lung_mask_path = lung_mask_path
        self.save_all_images = save_all_images
        self.batch_size = batch_size

    def model_inference(self):
        raise NotImplementedError()

    def run(self):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        target_studys, pred_studys = [], []
        for vol_idx, (raw_vol, vol, mask_vol, infos) in enumerate(self.volume_generator):
            if self.max_test_cases is not None:
                if vol_idx >= self.max_test_cases:
                    break

            # Model Inference
            pid, scan_idx = infos['pid'], infos['scan_idx']
            print(f'\n Volume {vol_idx} Patient {pid} Scan {scan_idx}')
            
            pred_vol = self.model_inference(vol)
            # pred_vol = pytorch_model_inference(self.predictor, self.data_converter)
            # pred_vol = d2_model_inference(vol, batch_size=cfg.TEST_BATCH_SIZE, predictor=predictor)

            # Data post-processing
            # TODO: the target volume should reduce small area but 1 pixel remain in 1m0037 131
            pred_vol_category = self.post_processer(pred_vol)
            # target_vol_category = post_processer.connect_components(mask_vol, connectivity=cfg.connectivity)
            target_vol_category = self.post_processer(mask_vol)

            # False positive reducing
            if self.fp_reducer is not None:
                pred_vol_category = self.fp_reducer(pred_vol_category, raw_vol, self.lung_mask_path, pid)

            # Nodule classification
            if self.nodule_classifier is not None:
                pred_vol_category, pred_nodule_info = self.nodule_classifier.nodule_classify(vol, pred_vol_category, mask_vol)
            else:
                pred_nodule_info = None

            # Evaluation
            target_study = LungNoduleStudy(pid, target_vol_category, raw_volume=raw_vol)
            pred_study = LungNoduleStudy(pid, pred_vol_category, raw_volume=raw_vol)
            self.eval_metrics.calculate(target_study, pred_study)

            # Visualize
            # # TODO: repeat part
            # origin_save_path = os.path.join(self.save_path, 'images', pid, 'origin')
            # enlarge_save_path = os.path.join(self.save_path, 'images', pid, 'enlarge')
            # _3d_save_path = os.path.join(self.save_path, 'images', pid, '3d')
            # for path in [origin_save_path, enlarge_save_path, _3d_save_path]:
            #     os.makedirs(path, exist_ok=True)
            if self.save_vis_condition(vol_idx):
                nodule_visualize(self.save_path, pid, vol, mask_vol, pred_vol, target_vol_category, pred_vol_category, 
                                 pred_nodule_info, self.save_all_images)

            target_studys.append(target_study)
            pred_studys.append(pred_study)

        _ = self.eval_metrics.evaluation(show_evaluation=True)
        return target_studys, pred_studys


class D2SegEvaluator(NoudleSegEvaluator):
    def __init__(self, predictor, volume_generator, save_path, data_converter, eval_metrics, slice_shift, save_vis_condition=True, 
                 max_test_cases=None, post_processer=None, fp_reducer=None, nodule_classifier=None, lung_mask_path='./', save_all_images=False, batch_size=1):
        super().__init__(predictor, volume_generator, save_path, data_converter, eval_metrics, save_vis_condition, 
                         max_test_cases, post_processer, fp_reducer, nodule_classifier, lung_mask_path, save_all_images, batch_size)
        self.slice_shift = slice_shift
    
    def model_inference(self, vol):
        pred_vol = np.zeros_like(vol[...,0])
        # dataset = self.data_converter(vol, slice_shift=self.slice_shift)
        # dataloder = DataLoader(dataset, batch_size=1, shuffle=False)
        for batch_start_index in range(0, vol.shape[0], self.batch_size):
        # for idx, input_data in enumerate(dataloder):
            start, end = batch_start_index, min(vol.shape[0], batch_start_index+self.batch_size)
            input_data = vol[start:end]
            img_list = np.split(input_data, input_data.shape[0], axis=0)
            outputs = self.predictor(img_list)
            for j, output in enumerate(outputs):
                pred = output["instances"]._fields['pred_masks'].cpu().detach().numpy() 
                pred = np.sum(pred, axis=0)
                pred = self.mask_preprocess(pred)
                pred_vol[batch_start_index+j] = pred
        return pred_vol

    @staticmethod
    def mask_preprocess(mask, ignore_malignancy=True, output_dtype=np.int32):
        # assert mask.ndim == 2
        if ignore_malignancy:
            mask = np.where(mask>=1, 1, 0)
        mask = output_dtype(mask)
        return mask

class Pytorch2dSegEvaluator(NoudleSegEvaluator):
    def __init__(self, predictor, volume_generator, save_path, data_converter, eval_metrics, slice_shift, save_vis_condition=True, 
                 max_test_cases=None, post_processer=None, fp_reducer=None, nodule_classifier=None, lung_mask_path='./', save_all_images=False, batch_size=1):
        super().__init__(predictor, volume_generator, save_path, data_converter, eval_metrics, save_vis_condition, 
                         max_test_cases, post_processer, fp_reducer, nodule_classifier, lung_mask_path, save_all_images, batch_size)
        self.slice_shift = slice_shift
    
    def model_inference(self, vol):
        vol = np.float32(vol[...,0])
        dataset = self.data_converter(vol, slice_shift=self.slice_shift)
        dataloder = DataLoader(dataset, batch_size=1, shuffle=False)
        for idx, input_data in enumerate(dataloder):
            input_data = input_data.to(torch.device('cuda:0'))
            pred = self.predictor(input_data)['out']
            pred = nn.Softmax(dim=1)(pred)
            pred = torch.argmax(pred, dim=1)
            if idx == 0:
                pred_vol = pred
            else:
                pred_vol = torch.cat([pred_vol, pred], 0)
        pred_vol = pred_vol.cpu().detach().numpy()
        return pred_vol


class Pytorch3dSegEvaluator(NoudleSegEvaluator):
    def __init__(self, predictor, volume_generator, save_path, data_converter, eval_metrics, save_vis_condition=True, 
                 max_test_cases=None, post_processer=None, fp_reducer=None, nodule_classifier=None, lung_mask_path='./', save_all_images=False, batch_size=1):
        super().__init__(predictor, volume_generator, save_path, data_converter, eval_metrics, save_vis_condition, 
                         max_test_cases, post_processer, fp_reducer, nodule_classifier, lung_mask_path, save_all_images, batch_size)
    
    def model_inference(self, vol):
        vol = np.float32(vol[...,0])
        vol = np.swapaxes(np.swapaxes(vol, 0, 2), 0, 1)
        vol /= 255
        dataset = self.data_converter(vol, crop_range=(64,64,32), crop_shift=(0,0,0))
        dataloder = DataLoader(dataset, batch_size=1, shuffle=False)
        for idx, input_data in enumerate(dataloder):
            input_data = input_data.to(torch.device('cuda:0'))
            pred = self.predictor(input_data)
            # pred = nn.Softmax(dim=1)(pred)
            # pred = torch.argmax(pred, dim=1)

            # input_data_np = input_data.cpu().detach().numpy()
            # dd = input_data_np[0,0,...,10]
            # if np.sum(dd)>0:
            #     plt.imshow(dd)
            #     plt.show()  
            # pred = nn.Sigmoid()(pred)

            pred = torch.where(pred>0.5, 1, 0)
            pred_np = pred.cpu().detach().numpy()
            if np.sum(pred_np)>0:
                print('!!!!!!')
                for s in range(32):
                    if np.sum(pred_np[...,s])>0:
                        plt.imshow(pred_np[0,0,...,s])
                        plt.show()
        # pred_vol = pred_vol.cpu().detach().numpy()
        pred_vol = np.zeros_like(vol)
        pred_vol = np.swapaxes(np.swapaxes(pred_vol, 0, 2), 1, 2)
        return pred_vol


def nodule_visualize(save_path, pid, vol, mask_vol, pred_vol, target_vol_category, pred_vol_category, pred_nodule_info, save_all_images=False):
    origin_save_path = os.path.join(save_path, 'images', pid, 'origin')
    enlarge_save_path = os.path.join(save_path, 'images', pid, 'enlarge')
    _3d_save_path = os.path.join(save_path, 'images', pid, '3d')
    for path in [origin_save_path, enlarge_save_path, _3d_save_path]:
        os.makedirs(path, exist_ok=True)

    vis_vol, vis_indices, vis_crops = visualize(vol, pred_vol_category, mask_vol, pred_nodule_info)
    if save_all_images:
        vis_indices = np.arange(vis_vol.shape[0])

    for vis_idx in vis_indices:
        # plt.savefig(vis_vol[vis_idx])
        cv2.imwrite(os.path.join(origin_save_path, f'vis-{pid}-{vis_idx}.png'), vis_vol[vis_idx])
        if vis_idx in vis_crops:
            for crop_idx, vis_crop in enumerate(vis_crops[vis_idx]):
                cv2.imwrite(os.path.join(enlarge_save_path, f'vis-{pid}-{vis_idx}-crop{crop_idx:03d}.png'), vis_crop)

    temp = np.where(mask_vol+pred_vol>0, 1, 0)
    zs_c, ys_c, xs_c = np.where(temp)
    crop_range = {'z': (np.min(zs_c), np.max(zs_c)), 'y': (np.min(ys_c), np.max(ys_c)), 'x': (np.min(xs_c), np.max(xs_c))}
    if crop_range['z'][1]-crop_range['z'][0] > 2 and \
       crop_range['y'][1]-crop_range['y'][0] > 2 and \
       crop_range['x'][1]-crop_range['x'][0] > 2:
        save_mask_in_3d(target_vol_category, 
                        save_path1=os.path.join(_3d_save_path, f'{pid}-raw-mask.png'),
                        save_path2=os.path.join(_3d_save_path, f'{pid}-preprocess-mask.png'), 
                        crop_range=crop_range)
        save_mask_in_3d(pred_vol_category,
                        save_path1=os.path.join(_3d_save_path, f'{pid}-raw-pred.png'),
                        save_path2=os.path.join(_3d_save_path, f'{pid}-preprocess-pred.png'),
                        crop_range=crop_range)