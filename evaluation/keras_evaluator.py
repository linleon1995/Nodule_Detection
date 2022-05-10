import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

from data.crop_utils import crops_to_volume
from utils.nodule import LungNoduleStudy
from data.volume_generator import asus_nodule_volume_generator
from visualization.vis import save_mask, visualize, save_mask_in_3d


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
            
            pred_vol = self.model_inference(vol, mask_vol)

            # Data post-processing
            # TODO: the target volume should reduce small area but 1 pixel remain in 1m0037 131
            pred_vol_category = self.post_processer(pred_vol)
            # target_vol_category = post_processer.connect_components(mask_vol, connectivity=cfg.connectivity)
            target_vol_category = self.post_processer(mask_vol)

            # for s in range(mask_vol.shape[0]):
            #     if np.sum(mask_vol[s])>0:
            #         plt.imshow(vol[s], 'gray')
            #         plt.imshow(mask_vol[s]+pred_vol[s]*2, alpha=0.2)
            #         # plt.savefig(f'plot/cube-{idx}-{s}.png')
            #         plt.show()

            
            # Evaluation
            target_study = LungNoduleStudy(pid, target_vol_category, raw_volume=raw_vol)
            pred_study = LungNoduleStudy(pid, pred_vol_category, raw_volume=raw_vol)
            # TODO
            self.eval_metrics.calculate(target_study, pred_study)

            # False positive reducing
            if self.fp_reducer is not None:
                pred_vol_category = self.fp_reducer(pred_study, raw_vol, self.lung_mask_path, pid)

            # Nodule classification
            if self.nodule_classifier is not None:
                pred_vol_category, pred_nodule_info = self.nodule_classifier.nodule_classify(vol, pred_vol_category, mask_vol)
            else:
                pred_nodule_info = None


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
            print(f'Warning: No {f} exist with key {keys}.') 
        else: 
            print(f'Warning: No {f} exist.') 
    return file_list


def get_samples(roots, cases):
    samples = []
    for root in roots:
        samples.extend(get_files(root, keys=cases))
    return samples


def load_data(input_samples, target_samples, remove_zeros):
    total_input_data, total_target = [], []
    for idx, (x_path, y_path) in enumerate(zip(input_samples, target_samples)):
        # print(idx)
        input_data = np.load(x_path)
        target = np.load(y_path)

        if remove_zeros:
            if not np.sum(target):
                continue
            
        input_data, target = np.swapaxes(np.swapaxes(input_data, 0, 1), 1, 2), np.swapaxes(np.swapaxes(target, 0, 1), 1, 2)
        input_data = input_data / 255
        input_data, target = input_data[np.newaxis], target[np.newaxis]

        total_input_data.append(input_data)
        total_target.append(target)

    total_input_data = np.concatenate(total_input_data, axis=0)
    total_target = np.concatenate(total_target, axis=0)
    return total_input_data, total_target


class Keras3dSegEvaluator(NoudleSegEvaluator):
    def __init__(self, predictor, volume_generator, save_path, data_converter, eval_metrics, save_vis_condition=True, 
                 max_test_cases=None, post_processer=None, fp_reducer=None, nodule_classifier=None, lung_mask_path='./', 
                 save_all_images=False, batch_size=1, *args, **kwargs):
        super().__init__(predictor, volume_generator, save_path, data_converter, eval_metrics, save_vis_condition, 
                         max_test_cases, post_processer, fp_reducer, nodule_classifier, lung_mask_path, save_all_images, batch_size)
    
    def get_sample(self):
        import os
        key = '32x64x64-10-shift-8'
        input_roots = [
                    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'positive', 'Image'),
                    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'positive', 'Image'),
                os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'positive', 'Image'),
                os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'positive', 'Image'),
                    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'negative', 'Image'),
                    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'negative', 'Image'),
                os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'negative', 'Image'),
                os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'negative', 'Image'),
                    ]
        target_roots = [
                    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'positive', 'Mask'),
                    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'positive', 'Mask'),
                    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'positive', 'Mask'),
                    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'positive', 'Mask'),
                    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'negative', 'Mask'),
                    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Malignant', 'crop', key, 'negative', 'Mask'),
                    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'negative', 'Mask'),
                    os.path.join(rf'C:\Users\test\Desktop\Leon\Datasets\ASUS_Nodules-preprocess\ASUS-Benign', 'crop', key, 'negative', 'Mask'),
                        ]
        
        train_file_keys = [f'1m{idx:04d}' for idx in range(1, 37)] + [f'1B{idx:04d}' for idx in range(1, 21)]
        valid_file_keys = [f'1m{i:04d}' for i in range(37, 39)] + [f'1B{idx:04d}' for idx in range(21, 23)]
        test_file_keys = [f'1m{i:04d}' for i in range(37, 45)] + [f'1B{idx:04d}' for idx in range(21, 26)]

        test_input_samples = get_samples(input_roots, test_file_keys)   
        test_target_samples = get_samples(target_roots, test_file_keys) 
        x_test, y_test = load_data(test_input_samples, test_target_samples, remove_zeros=False)
        x_test = x_test[:,np.newaxis]
        y_test = y_test[:,np.newaxis]
        x_test = x_test[:24]
        y_test = y_test[:24]
        return x_test, y_test

    def model_inference(self, vol, mask_vol):
        # x_crops, y_crops = self.get_sample()
        
        # TODO:
        crop_range, crop_shift, convert_dtype, overlapping = [64,64,32], [0,0,0], np.float32, 1.0
        cropping_op = self.data_converter(crop_range, crop_shift, convert_dtype, overlapping)

        vol = vol[...,0] / 255
        vol = np.swapaxes(np.swapaxes(vol, 0, 2), 0, 1)
        crop_infos = cropping_op(vol)
        crop_data_list = [info['data'][np.newaxis,np.newaxis] for info in crop_infos]

        crop_slice_list = []
        for info in crop_infos:
            slice_list = []
            for slice_range in info['slice']:
                slice_list.append(slice(*slice_range))
                # slice_list.append(np.arange(*slice_range))
            crop_slice_list.append(slice_list)

        x_crops = np.concatenate(crop_data_list, axis=0)
        # TODO:
        # x_crops = x_crops[:24]
        print('x_crops: {} | {} ~ {}'.format(x_crops.shape, np.min(x_crops), np.max(x_crops)))

        pred_crops = self.predictor.predict(x_crops, verbose=1, batch_size=6)
        pred_vol = crops_to_volume(pred_crops, crop_slice_list, vol.shape)
        pred_vol = np.swapaxes(np.swapaxes(pred_vol, 0, 2), 1, 2)
        class_vol = np.where(pred_vol>0.5, 1, 0)

        # class_crops = np.where(pred_crops>0.5, 1, 0)
        # for i, p in enumerate(class_crops):
        #     for s in range(32):
        #         if np.sum(p[...,s])>0:
        #             print(i, np.sum(p))
        #             plt.imshow(x_crops[i,0,...,s], 'gray')
        #             plt.imshow(p[0,...,s], alpha=0.2)
        #             plt.savefig(f'plot/keras/pred/img_{i:03d}_{s:03d}.png')
        # return 0
        return class_vol