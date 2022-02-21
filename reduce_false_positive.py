import numpy as np
import collections
import torch
from data.luna16_data_preprocess import LUNA16_CropRange_Builder
from model.ResNet_3d import build_3d_resnet
from utils.volume_eval import volumetric_data_eval
from utils.utils import compare_result, compare_result_enlarge
from modules.utils import configuration
import matplotlib.pyplot as plt

# TODO: torch, numpy problem
class False_Positive_Reducer():
    def __init__(self, crop_range, checkpint_path, model_depth=50, num_class=2):
        self.crop_range = crop_range
        self.device = configuration.get_device()
        self.classifier = self.build_up_classifier(checkpint_path, model_depth, num_class)

    def reduce_false_positive(self, raw_volume, pred_volume):
        pred_volume, _ = volumetric_data_eval.volume_preprocess(pred_volume, connectivity=26, area_threshold=20)
        pred_nodules = self.get_nodule_center(pred_volume)
        for nodule in pred_nodules:
            # fig, ax = compare_result(raw_volume[nodule['Center']['index']], pred_volume[nodule['Center']['index']], pred_volume[nodule['Center']['index']])
            # fig2, ax2 = compare_result_enlarge(raw_volume[nodule['Center']['index']], pred_volume[nodule['Center']['index']], pred_volume[nodule['Center']['index']])
            # nodule_index = nodule['Center']['index']
            # fig.savefig(f'plot/fpr/{nodule_index:03d}.png')
            # fig2.savefig(f'plot/fpr/{nodule_index:03d}-en.png')
            crop_raw_volume = LUNA16_CropRange_Builder.crop_volume(raw_volume, self.crop_range, nodule['Center'])
            crop_raw_volume = np.swapaxes(np.expand_dims(crop_raw_volume, (0, 1)), 1, 5)[...,0]
            crop_raw_volume = torch.from_numpy(crop_raw_volume).float().to(self.device)

            # if nodule['Center']['index'] == 68:
            #     crop_pred_volume = LUNA16_CropRange_Builder.crop_volume(pred_volume, self.crop_range, nodule['Center'])
            #     # crop_pred_volume = np.swapaxes(np.expand_dims(crop_raw_volume, (0, 1)), 1, 5)[...,0]
            #     for j in range(48):
            #         if np.sum(crop_pred_volume[j]) > 0:
            #             plt.title(str(j))
            #             plt.imshow(crop_raw_volume.cpu().detach().numpy()[0,0,j], 'gray')
            #             plt.imshow(crop_pred_volume[j], alpha=0.2)
            #             plt.show()

            pred_class = self.inference(crop_raw_volume)
            if pred_class.item() == 0:
                pred_volume[pred_volume==nodule['Nodule_label']] = 0
        pred_volume[pred_volume>0] = 1
        return pred_volume

    def inference(self, input_volume):
        logits = self.classifier(input_volume)
        prob = torch.nn.functional.softmax(logits, dim=1)
        return torch.argmax(prob)

    def build_up_classifier(self, checkpint_path, model_depth, num_class):
        state_key = torch.load(checkpint_path, map_location=self.device)
        model = build_3d_resnet(model_depth, n_classes=num_class, conv1_t_size=7, conv1_t_stride=2)
        model.load_state_dict(state_key['net'])
        model = model.to(self.device)
        return model

    @classmethod
    def get_nodule_center(cls, pred_volume):
        total_nodule_center = []
        for label in np.unique(pred_volume)[1:]:
            zs, ys, xs = np.where(pred_volume==label)
            center_index, center_row, center_column = np.mean(zs), np.mean(ys), np.mean(xs)
            total_nodule_center.append({'Nodule_label': label, 
                                        'Center': {'index': np.mean(center_index).astype('int32'), 'row': np.mean(center_row).astype('int32'), 'column': np.mean(center_column).astype('int32')}})

        return total_nodule_center





