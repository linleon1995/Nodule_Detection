import os
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

from utils.nodule import LungNoduleStudy
from utils.utils import DataFrameTool
from visualization.vis import plot_image_truth_prediction, visualize, save_mask_in_3d
from data.crop_utils import crops_to_volume
from utils.metrics import binary_dsc

# TODO:
###
import logging


def save_in_nrrd(vol, mask_vol, direction, origin, spacing, save_dir, filename):
    from utils.slicer_utils import seg_nrrd_write, raw_nrrd_write
    vol = vol[...,0]
    vol = np.transpose(vol, (1, 2, 0))
    mask_vol = np.transpose(mask_vol, (1, 2, 0))
    spacing = np.array([spacing[1], spacing[2], spacing[0]])
    direction = direction * np.tile(spacing[:,np.newaxis], (1, 3))
    origin = np.array([origin[1], origin[2], origin[0]])

    # To fit the coordinate
    vol = np.rot90(vol, k=1)
    mask_vol = np.rot90(mask_vol, k=1)
    
    # plt.imshow(vol[...,10])
    # plt.show()
    raw_path = os.path.join(save_dir, f'{filename}.nrrd')
    seg_path = os.path.join(save_dir, f'{filename}.seg.nrrd')
    raw_nrrd_write(raw_path, vol, direction, origin, spacing)
    seg_nrrd_write(seg_path, mask_vol, direction, origin, spacing)


def get_files(path, keys=[], return_fullpath=True, sort=True, sorting_key=None, recursive=True, get_dirs=False, ignore_suffix=False):
    """Get all the file name under the given path with assigned keys
    Args:
        path: (str)
        keys: (list, str)
        return_fullpath: (bool)
        sort: (bool)
        sorting_key: (func)
        recursive: The flag for searching path recursively or not(bool)
    Return:
        file_list: (list)
    """
    file_list = []
    assert isinstance(keys, (list, str))
    if isinstance(keys, str): keys = [keys]
    # Rmove repeated keys
    keys = list(set(keys))

    def push_back_filelist(root, f, file_list, is_fullpath):
        f = f[:-4] if ignore_suffix else f
        if is_fullpath:
            file_list.append(os.path.join(root, f))
        else:
            file_list.append(f)

    for i, (root, dirs, files) in enumerate(os.walk(path)):
        # print(root, dirs, files)
        if not recursive:
            if i > 0: break

        if get_dirs:
            files = dirs
            
        for j, f in enumerate(files):
            if keys:
                for key in keys:
                    if key in f:
                        push_back_filelist(root, f, file_list, return_fullpath)
            else:
                push_back_filelist(root, f, file_list, return_fullpath)

    if file_list:
        if sort: file_list.sort(key=sorting_key)
    else:
        f = 'dir' if get_dirs else 'file'
        if keys: 
            logging.warning(f'No {f} exist with key {keys}.') 
        else: 
            logging.warning(f'No {f} exist.') 
    return file_list


def get_samples(roots, cases):
    samples = []
    for root in roots:
        samples.extend(get_files(root, keys=cases))
    return samples


def load_data(input_samples, target_samples, remove_zeros):
    total_input_data, total_target = [], []
    pid_dict = {}
    for idx, (x_path, y_path) in enumerate(zip(input_samples, target_samples)):
        # print(idx)
        input_data = np.load(x_path)
        target = np.load(y_path)

        if remove_zeros:
            if not np.sum(target):
                continue

        # if remove_shift:
        #     file_name = os.path.split(x_path)[1][:-4]
        #     pid = file_name.split('-')[-1]
        #     case_num = int(file_name.split('-')[1])
        #     if pid in pid_dict:
        #         continue
        #     else:
        #         pid_dict[pid] = case_num

        input_data = np.transpose(input_data, (1, 2, 0))
        target = np.transpose(target, (1, 2, 0))
        input_data = input_data / 255
        input_data, target = input_data[np.newaxis], target[np.newaxis]

        total_input_data.append(input_data)
        total_target.append(target)

    total_input_data = np.concatenate(total_input_data, axis=0)
    total_target = np.concatenate(total_target, axis=0)
    return total_input_data, total_target
###
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
        # TODO:
        self.quick_test = False
        self.remove_result = {}

    def model_inference(self):
        raise NotImplementedError()

    def reduce_fp(self, pred_study):
        total_removal = []
        for name, remove_nodule_ids in pred_study.remove_nodule_record.items():
            other_method = pred_study.remove_nodule_record.copy()
            other_method.pop(name)
            other_method_pred = []
            for idx in other_method.values():
                other_method_pred.extend(list(idx))
            unique_pred = list(set(remove_nodule_ids)-set(other_method_pred))
            unique_pred_num = len(unique_pred)
            common_pred_num = len(remove_nodule_ids) - unique_pred_num
            if name not in self.remove_result:
                self.remove_result[name] = {}


            if 'unique' in self.remove_result[name] and 'common' in self.remove_result[name]:
                self.remove_result[name]['unique'] = self.remove_result[name]['unique'] + unique_pred_num
                self.remove_result[name]['common'] = self.remove_result[name]['common'] + common_pred_num
            else:
                self.remove_result[name].update({'unique': unique_pred_num, 'common': common_pred_num})

            tp, fp = 0, 0
            total_removal.extend(remove_nodule_ids)
            for n_id in remove_nodule_ids:
                if 'tp' in pred_study.nodule_evals[n_id]:
                    tp += 1
                elif 'fp' in pred_study.nodule_evals[n_id]:
                    fp += 1
            print(f'-- {name} TP {tp} FP {fp}')
            if 'TP' in self.remove_result[name] and 'FP' in self.remove_result[name]:
                self.remove_result[name]['TP'] = self.remove_result[name]['TP'] + tp
                self.remove_result[name]['FP'] = self.remove_result[name]['FP'] + fp
            else:
                self.remove_result[name].update({'TP': tp, 'FP': fp})

        total_removal = list(set(total_removal))
        post_nodule_num = len(pred_study.nodule_evals)-len(total_removal)
        

        pred_vol_category = pred_study.category_volume
        tp, fp = 0, 0
        for remove_nodule_id in total_removal:
            if 'tp' in pred_study.nodule_evals[remove_nodule_id]:
                tp += 1
            elif 'fp' in pred_study.nodule_evals[remove_nodule_id]:
                fp += 1
            pred_vol_category[pred_vol_category==remove_nodule_id] = 0
        pred_study.category_volume = pred_vol_category
        if 'total' in self.remove_result:
            self.remove_result['total']['TP'] = self.remove_result['total']['TP'] + tp
            self.remove_result['total']['FP'] = self.remove_result['total']['FP'] + fp
        else:
            self.remove_result['total'] = {'TP': tp, 'FP': fp}
        return pred_study, post_nodule_num
    
    def run(self):
        if not os.path.isdir(self.save_path):
            os.makedirs(self.save_path)

        target_studys, pred_studys = [], []
        total_psot_nodule_Num = 0
        total_eval_time = 0
        case_eval = DataFrameTool(
            ['pid', 'inference_time', 'pred_nodule_num', 'tp', 'fp', 'fn', 'double_detect'])
        for vol_idx, (raw_vol, vol, mask_vol, infos) in enumerate(self.volume_generator):
            # if vol.shape[0]<=500:
            #     continue
            one_case_eval = []
            if self.max_test_cases is not None:
                if vol_idx >= self.max_test_cases:
                    break

            # Model Inference
            pid, scan_idx = infos['pid'], infos['scan_idx']
            print(f'\n Volume {vol_idx} Patient {pid} Scan {scan_idx}')
            
            inference_start = time.time()
            pred_vol = self.model_inference(vol)
            inference_end = time.time()
            inference_time = inference_end - inference_start
            print(f'Inference time {inference_time:.2f}')

            if self.quick_test:
                pred_vol *= mask_vol

            # Data post-processing
            pred_vol_category = self.post_processer(pred_vol)
            target_vol_category = self.post_processer(mask_vol)
            num_pred_nodule = np.unique(pred_vol_category).size-1
 
            # Evaluation
            eval_start = time.time()
            target_study = LungNoduleStudy(pid, target_vol_category, raw_volume=raw_vol)
            pred_study = LungNoduleStudy(pid, pred_vol_category, raw_volume=raw_vol)
            tp, fp, fn, doubleCandidatesIgnored = self.eval_metrics.calculate(target_study, pred_study)
            eval_end = time.time()
            eval_time = eval_end - eval_start
            print(f'Eval time {eval_time:.2f}')
            total_eval_time += eval_time


            print(f'Predict Nodules (raw) {num_pred_nodule}')

            # nrrd_path = os.path.join(self.save_path, 'images', pid, 'nrrd')
            # os.makedirs(nrrd_path, exist_ok=True)
            # save_in_nrrd(vol, pred_vol_category, direction, origin, spacing, nrrd_path, pid)

            pred_nodule_info = None
            if self.fp_reducer is not None or self.nodule_classifier is not None:
                # False positive reducing
                if self.fp_reducer is not None:
                    pred_study = self.fp_reducer(pred_study, raw_vol, self.lung_mask_path, pid)

                # Nodule classification
                if self.nodule_classifier is not None:
                    pred_study, pred_nodule_info = self.nodule_classifier.nodule_classify(vol, pred_study, mask_vol)
                # else:
                #     pred_nodule_info = None
            
                pred_study, post_nodule_num = self.reduce_fp(pred_study)
                print(f'Predict Nodules (post) {post_nodule_num}')
                total_psot_nodule_Num += post_nodule_num
                pred_vol_category = pred_study.category_volume

            # Visualize
            if self.save_vis_condition(vol_idx):
                nodule_visualize(self.save_path, pid, vol, mask_vol, pred_vol, target_vol_category, pred_vol_category, 
                                 pred_nodule_info, self.save_all_images)

            # target_studys.append(target_study)
            # pred_studys.append(pred_study)
            one_case_eval = [pid, inference_time, num_pred_nodule, tp, fp, fn, doubleCandidatesIgnored]
            case_eval.write_row(one_case_eval)

        eval_start = time.time()
        _ = self.eval_metrics.evaluation(show_evaluation=True)
        eval_end = time.time()
        eval_time = eval_end - eval_start
        total_eval_time += eval_time
        print(f'Total Eval time {total_eval_time:.2f}')

        case_eval.save_data_frame(os.path.join(self.save_path, 'case_eval.csv'))
        print(f'Total post candidates {total_psot_nodule_Num}')
        print(self.remove_result)

        common_pred, unique_pred = [], []
        for removal_info in self.remove_result.values():
            if 'unique' in removal_info and 'common' in removal_info:
                common_pred.append(removal_info['common'])
                unique_pred.append(removal_info['unique'])
        remove_method = list(self.remove_result.keys())
        remove_method.remove('total')
        
        ind = np.arange(len(self.remove_result)-1) # the x locations for the groups
        width = 0.35
        fig, ax = plt.subplots(1, 1)
        ax.bar(ind, common_pred, width, color='b')
        ax.bar(ind, unique_pred, width, bottom=common_pred, color='r')
        ax.set_xlabel('Method')
        ax.set_ylabel('Nodule Num.')
        ax.set_title('Comparision of unique FP removal')
        ax.set_xticks(ind)
        ax.set_xticklabels(remove_method)
        ax.set_yticks(np.arange(0, max(unique_pred)+max(common_pred), (max(unique_pred)+max(common_pred))//10))
        ax.legend(labels=['Common', 'Unique'])
        fig.savefig(os.path.join(self.save_path, 'unique_FP_removal.png'))
        # plt.show()
        # TODO:
        target_studys = [target_study]
        pred_studys = [pred_study]
        return target_studys, pred_studys


class D2SegEvaluator(NoudleSegEvaluator):
    def __init__(self, predictor, volume_generator, save_path, data_converter, eval_metrics, slice_shift, save_vis_condition=True, 
                 max_test_cases=None, post_processer=None, fp_reducer=None, nodule_classifier=None, lung_mask_path='./', 
                 save_all_images=False, batch_size=4, *args, **kwargs):
        super().__init__(predictor, volume_generator, save_path, data_converter, eval_metrics, save_vis_condition, 
                         max_test_cases, post_processer, fp_reducer, nodule_classifier, lung_mask_path, save_all_images, batch_size)
        self.slice_shift = slice_shift
    
    def model_inference(self, vol):
        pred_vol = np.zeros_like(vol[...,0])
        for batch_start_index in range(0, vol.shape[0], self.batch_size):
        # for idx, input_data in enumerate(dataloder):
            start, end = batch_start_index, min(vol.shape[0], batch_start_index+self.batch_size)
            input_data = vol[start:end]
            img_list = np.split(input_data, input_data.shape[0], axis=0)
            outputs = self.predictor(img_list)
            for j, output in enumerate(outputs):
                pred = output["instances"]._fields['pred_masks'].cpu().detach().numpy() 

                # pred_classes = output["instances"]._fields['pred_classes'].cpu().detach().numpy() 
                # pred_classes = np.reshape(pred_classes, (pred_classes.size, 1, 1))

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
                 max_test_cases=None, post_processer=None, fp_reducer=None, nodule_classifier=None, lung_mask_path='./', 
                 save_all_images=False, batch_size=1, *args, **kwargs):
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
                 max_test_cases=None, post_processer=None, fp_reducer=None, nodule_classifier=None, lung_mask_path='./', 
                 save_all_images=False, batch_size=1, overlapping=0.5, reweight=True, reweight_sigma=0.25, *args, **kwargs):
        super().__init__(predictor, volume_generator, save_path, data_converter, eval_metrics, save_vis_condition, 
                         max_test_cases, post_processer, fp_reducer, nodule_classifier, lung_mask_path, save_all_images, batch_size)
        
        self.overlapping = overlapping
        self.reweight = reweight
        self.reweight_sigma = reweight_sigma

    def crop_test(self, vol, mask_vol):
        # key = '32x64x64-10-shift-8'
        remove_zeros = False
        key = '32x64x64-10'
        input_roots = [
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Malignant', 'crop', key, 'positive', 'Image'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Benign', 'crop', key, 'positive', 'Image'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Malignant', 'crop', key, 'negative', 'Image'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Benign', 'crop', key, 'negative', 'Image'),
        ]
        target_roots = [
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Malignant', 'crop', key, 'positive', 'Mask'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Benign', 'crop', key, 'positive', 'Mask'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Malignant', 'crop', key, 'negative', 'Mask'),
            os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\TMH-Benign', 'crop', key, 'negative', 'Mask'),
        ]

        train_file_keys = [f'1m{idx:04d}' for idx in range(1, 37)] + [f'1B{idx:04d}' for idx in range(1, 21)]
        valid_file_keys = [f'1m{i:04d}' for i in range(37, 39)] + [f'1B{idx:04d}' for idx in range(21, 23)]
        test_file_keys = [f'1m{i:04d}' for i in range(37, 45)] + [f'1B{idx:04d}' for idx in range(21, 26)]


        train_input_samples = get_samples(input_roots, train_file_keys)   
        train_target_samples = get_samples(target_roots, train_file_keys) 
        x_train, y_train = load_data(train_input_samples, train_target_samples, remove_zeros=remove_zeros)
        x_train = x_train[:,np.newaxis]
        y_train = y_train[:,np.newaxis]

        valid_input_samples = get_samples(input_roots, valid_file_keys)   
        valid_target_samples = get_samples(target_roots, valid_file_keys) 
        x_valid, y_valid = load_data(valid_input_samples, valid_target_samples, remove_zeros=remove_zeros)
        x_valid = x_valid[:,np.newaxis]
        y_valid = y_valid[:,np.newaxis]

        test_input_samples = get_samples(input_roots, test_file_keys)   
        test_target_samples = get_samples(target_roots, test_file_keys) 
        x_test, y_test = load_data(test_input_samples, test_target_samples, remove_zeros=remove_zeros)
        x_test = x_test[:,np.newaxis]
        y_test = y_test[:,np.newaxis]

        print('x_train: {} | {} ~ {}'.format(x_train.shape, np.min(x_train), np.max(x_train)))
        print('y_train: {} | {} ~ {}'.format(y_train.shape, np.min(y_train), np.max(y_train)))

        print('x_valid: {} | {} ~ {}'.format(x_valid.shape, np.min(x_valid), np.max(x_valid)))
        print('y_valid: {} | {} ~ {}'.format(y_valid.shape, np.min(y_valid), np.max(y_valid)))

        print('x_test: {} | {} ~ {}'.format(x_test.shape, np.min(x_test), np.max(x_test)))
        print('y_test: {} | {} ~ {}'.format(y_test.shape, np.min(y_test), np.max(y_test)))

        x_data, y_data = x_test, y_test
        x_data_p = torch.from_numpy(x_data).float().cuda()
        
        p_data = []
        for i in range(x_data.shape[0]):
            p = self.predictor(x_data_p[i:i+1])
            p_data.append(p.cpu().detach().numpy())

        p_data = np.concatenate(p_data, axis=0)
        print('CWD', os.getcwd())
        for i in range(0, x_data.shape[0], 1):
            plot_image_truth_prediction(
                x_data[i], y_data[i], p_data[i], rows=5, cols=5, name=f'plot/pytorch/img{i:03d}.png')

    def crop_test_luna16(self):
        remove_zeros = False

        input_roots = [
            rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop\32x64x64-1\positive\Image',
        ]
        target_roots = [
            rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16-preprocess\crop\32x64x64-1\positive\Mask',
        ]

        train_file_keys = [f'luna16-{idx:04d}' for idx in range(852)]
        valid_file_keys = [f'luna16-{idx:04d}' for idx in range(852, 963)]
        # test_file_keys = [f'luna16-{idx:04d}' for idx in range(963, 1186)]
        test_file_keys = [f'luna16-{idx:04d}' for idx in range(112)]
        # train_file_keys = [f'luna16-{idx:04d}' for idx in range(112)]
        # valid_file_keys = [f'luna16-{idx:04d}' for idx in range(112, 240)]
        # test_file_keys = [f'luna16-{idx:04d}' for idx in range(112, 240)]


        # train_input_samples = get_samples(input_roots, train_file_keys)   
        # train_target_samples = get_samples(target_roots, train_file_keys) 
        # x_train, y_train = load_data(train_input_samples, train_target_samples, remove_zeros=remove_zeros)
        # x_train = x_train[:,np.newaxis]
        # y_train = y_train[:,np.newaxis]

        # valid_input_samples = get_samples(input_roots, valid_file_keys)   
        # valid_target_samples = get_samples(target_roots, valid_file_keys) 
        # x_valid, y_valid = load_data(valid_input_samples, valid_target_samples, remove_zeros=remove_zeros)
        # x_valid = x_valid[:,np.newaxis]
        # y_valid = y_valid[:,np.newaxis]

        test_input_samples = get_samples(input_roots, test_file_keys)   
        test_target_samples = get_samples(target_roots, test_file_keys) 
        x_test, y_test = load_data(test_input_samples, test_target_samples, remove_zeros=remove_zeros)
        x_test = x_test[:,np.newaxis]
        y_test = y_test[:,np.newaxis]

        # print('x_train: {} | {} ~ {}'.format(x_train.shape, np.min(x_train), np.max(x_train)))
        # print('y_train: {} | {} ~ {}'.format(y_train.shape, np.min(y_train), np.max(y_train)))

        # print('x_valid: {} | {} ~ {}'.format(x_valid.shape, np.min(x_valid), np.max(x_valid)))
        # print('y_valid: {} | {} ~ {}'.format(y_valid.shape, np.min(y_valid), np.max(y_valid)))

        print('x_test: {} | {} ~ {}'.format(x_test.shape, np.min(x_test), np.max(x_test)))
        print('y_test: {} | {} ~ {}'.format(y_test.shape, np.min(y_test), np.max(y_test)))

        x_data, y_data = x_test, y_test
        x_data_p = torch.from_numpy(x_data).float().cuda()
        
        p_data = []
        for i in range(x_data.shape[0]):
            p = self.predictor(x_data_p[i:i+1])
            p_data.append(p.cpu().detach().numpy())

        p_data = np.concatenate(p_data, axis=0)
        print('CWD', os.getcwd())
        total_dsc = []
        crop_save_path = os.path.join(self.save_path, 'images', 'crop')
        os.makedirs(crop_save_path, exist_ok=True)
        for i in range(0, x_data.shape[0], 1):
            dsc = binary_dsc(y_data[i], p_data[i])
            total_dsc.append(dsc)
            # if i== 70:
            #     for j in range(32):
            #         show_mask_base(x_data[i,0,...,j], y_data[i,0,...,j], save_path=f'plot/test{i}_{j}.png')
            if i%5==0:
                plot_image_truth_prediction(
                    x_data[i], y_data[i], p_data[i], rows=5, cols=5, name=os.path.join(crop_save_path, f'img{i:03d}.png'))
        mean_dsc = sum(total_dsc)/len(total_dsc)
        print(f'Mean cropping DSC {mean_dsc:.4f}')

    def model_inference(self, vol):
        # TODO: remove mask_vol
        vol = np.float32(vol[...,0])
        vol = np.transpose(vol, (1, 2, 0))
        # mask_vol = np.transpose(mask_vol, (1, 2, 0))
        vol /= 255

        pred_vol = torch.zeros(vol.shape)
        # TODO: dataloader takes long time
        dataset = self.data_converter(vol, (64,64,32), (0,0,0), overlapping=self.overlapping)
        dataloder = DataLoader(dataset, batch_size=4, shuffle=False)
        pred_crops, pred_slices = [], []
        for idx, input_data in enumerate(dataloder):
            # if idx%20!=0:
            #     continue
            data, bbox = input_data['data'], input_data['slice']
            
            # data = data.float()
            
            # # TODO: custom bbox
            # bbox = [slice(256+32, 320+32), slice(0+32, 64+32), slice(32, 64)]
            # data = vol[bbox][np.newaxis,np.newaxis]
            # data = torch.from_numpy(data)

            data = data.to(torch.device('cuda:0'))
            pred = self.predictor(data)
            # print(data, bbox, bbox[0])
            # bbox = [slice(*tuple(torch.tensor(s).cpu().detach().numpy())) for s in bbox]
            batch_slices = []
            for dim, dim_slice in enumerate(bbox):
                start_indices = dim_slice[0].cpu().detach().numpy()
                end_indices = dim_slice[1].cpu().detach().numpy()
                for batch_idx, (start_idx, end_idx) in enumerate(zip(start_indices, end_indices)):
                    if dim == 0:
                        batch_slices.append([slice(start_idx, end_idx)])
                    else:
                        batch_slices[batch_idx].append(slice(start_idx, end_idx))

            # pred_slices.append(bbox)
            pred_slices.extend(batch_slices)

            pred_np = pred.cpu().detach().numpy()
            pred_crops.append(pred_np)

        pred_crops = np.concatenate(pred_crops, axis=0)
        pred_vol = crops_to_volume(pred_crops, pred_slices, vol.shape, self.reweight, self.reweight_sigma)
        pred_vol = np.where(pred_vol>0.5, 1, 0)
        pred_vol = np.transpose(pred_vol, (2, 0, 1))
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
    if np.sum(temp) > 0:
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

