import os
from model.ResNet_3d import build_3d_resnet
from torch.utils.data import Dataset, DataLoader
from nodule_classification.data.data_loader import Luna16CropDataset
from nodule_classification.data.data_loader import ASUSCropDataset
import numpy as np
import torch
from pprint import pprint
import tensorboardX
# from data.luna16_crop_preprocess import LUNA16_CropRange_Builder
# from data.asus_crop_preprocess import ASUS_CropRange_Builder
from nodule_classification.data.data_loader import BaseCropClsDataset
from utils.utils import build_size_figure

from utils.configuration import load_config, get_device
from utils import metrics
from utils.train_utils import get_logger, DictAsMember, create_activation
from data.data_utils import get_pids_from_coco

CONFIG_PATH = 'config_file/test_config.yml'
LOGGER = get_logger('test')


def load_checkpoint(checkpoint_path, model, optimizer=None,
                    model_key='model_state_dict', optimizer_key='optimizer_state_dict'):
    """Loads model and training parameters from a given checkpoint_path
    If optimizer is provided, loads optimizer's state_dict of as well.
    Args:
        checkpoint_path (string): path to the checkpoint to be loaded
        model (torch.nn.Module): model into which the parameters are to be copied
        optimizer (torch.optim.Optimizer) optional: optimizer instance into
            which the parameters are to be copied
    Returns:
        state
    """
    if not os.path.exists(checkpoint_path):
        raise IOError(f"Checkpoint '{checkpoint_path}' does not exist")

    state = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state[model_key])

    if optimizer is not None:
        optimizer.load_state_dict(state[optimizer_key])

    return state


class Evaluator():
    def __init__(self, 
                 cfg,
                model,
                test_dataloader,
                logger,
                device,
                activation_func=None,
                USE_TENSORBOARD=True,
                USE_CUDA=True):
        self.cfg = cfg
        self.model = model
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.iterations = 0
        self.device = device
        self.activation_func = activation_func
        self.USE_TENSORBOARD = USE_TENSORBOARD
        self.checkpoint_path = self.cfg.CHECKPOINT_PATH
        
    def evaluate(self):
        self.model.eval()
        self.eval_tool = metrics.SegmentationMetrics(self.cfg.MODEL.NUM_CLASSES, ['accuracy'])
        valid_samples = len(self.test_dataloader.dataset)
        for idx, data in enumerate(self.test_dataloader):
            input_var, labels = data['input'], data['target']
            labels = labels.long()
            input_var, labels = input_var.to(self.device), labels.to(self.device)
            outputs = self.model(input_var)
            
            if self.activation_func:
                if self.cfg.MODEL.ACTIVATION == 'softmax':
                    prob = self.activation_func(outputs)
                else:
                    prob = self.activation_func(outputs)
            else:
                prob = outputs
            prediction = torch.argmax(prob, dim=1)
            labels = labels[:,0]

            labels = labels.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            evals = self.eval_tool(labels, prediction)

        self.avg_test_acc = metrics.accuracy(
                np.sum(self.eval_tool.total_tp), np.sum(self.eval_tool.total_fp), np.sum(self.eval_tool.total_fn), np.sum(self.eval_tool.total_tn)).item()
        precision = metrics.precision(np.sum(self.eval_tool.total_tp), np.sum(self.eval_tool.total_fp))
        recall = metrics.recall(np.sum(self.eval_tool.total_tp), np.sum(self.eval_tool.total_fn))
        specificity = metrics.specificity(np.sum(self.eval_tool.total_tn), np.sum(self.eval_tool.total_fp))
        print(f'Precision: {precision*100:.02f}')
        print(f'Recall: {recall*100:.02f}')
        print(f'Specificity: {specificity*100:.02f}')
        print('Acc', self.avg_test_acc)
        print(f'TP: {self.eval_tool.total_tp}', 
              f'FP: {self.eval_tool.total_fp}',
              f'TN: {self.eval_tool.total_tn}',
              f'FN: {self.eval_tool.total_fn}')
    

def main(config_reference):
    # Configuration
    cfg = load_config(config_reference)
    cfg = DictAsMember(cfg)
    device = get_device(cfg.get('device', None))
    cfg.device = device
    pprint(cfg)

    model = build_3d_resnet(model_depth=cfg.MODEL.DEPTH, n_classes=cfg.MODEL.NUM_CLASSES, conv1_t_size=7, conv1_t_stride=2)
    state_key = torch.load(cfg.CHECKPOINT_PATH, map_location=cfg.device)
    model.load_state_dict(state_key['net'])
    model = model.to(cfg.device)

    test_seriesuid = get_pids_from_coco(
        os.path.join(cfg.DATA.COCO_PATH[cfg.DATA.NAME], 
                    # cfg.TRAIN.TASK_NAME, f'cv-{cfg.CV.FOLD}', str(cfg.CV.ASSIGN), 
                    'annotations_test.json'))
    test_dataset = BaseCropClsDataset(cfg.DATA.DATA_PATH[cfg.DATA.NAME], cfg.DATA.CROP_RANGE, 
                                      test_seriesuid, data_augmentation=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    # test_datasets = []
    # for dataset_name in cfg.DATA.NAME:
        
    #     if dataset_name == 'LUNA16':
    #         file_name_key = LUNA16_CropRange_Builder.get_filename_key(cfg.DATA.CROP_RANGE, cfg.DATA.NPratio)
    #         data_path = os.path.join(cfg.DATA.DATA_PATH[dataset_name], file_name_key)
    #         test_dataset = Luna16CropDataset(data_path, cfg.DATA.CROP_RANGE, mode='test')
    #     elif dataset_name in ['ASUS-B', 'ASUS-M']:
    #         file_name_key = ASUS_CropRange_Builder.get_filename_key(cfg.DATA.CROP_RANGE, cfg.DATA.NPratio)
    #         data_path = os.path.join(cfg.DATA.DATA_PATH[dataset_name], file_name_key)
    #         test_dataset = ASUSCropDataset(
    #             data_path, cfg.DATA.CROP_RANGE, negative_to_positive_ratio=cfg.DATA.NPratio_test, 
    #             nodule_type=dataset_name, mode='test')

    #     print(f'Dataset: {cfg.DATA.NAME} Test number: {len(test_dataset)}')
    #     test_datasets.append(test_dataset)

    # total_test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    # test_dataloader = DataLoader(total_test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    

    # # test_dataset = Luna16CropDataset(cfg.DATA.DATA_PATH, cfg.DATA.CROP_RANGE, mode='test')
    # # test_dataset = ASUSCropDataset(cfg.DATA.DATA_PATH, cfg.DATA.CROP_RANGE, nodule_type='ASUS-B', mode='test')
    # test_dataset = ASUSCropDataset(cfg.DATA.DATA_PATH, cfg.DATA.CROP_RANGE, nodule_type='ASUS-M', mode='test')
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    # Logger
    LOGGER.info("Start Evaluation!!")
    LOGGER.info("Batch size: {} Shuffling Data: {} Testing Samples: {}".
            format(cfg.DATA.BATCH_SIZE, cfg.DATA.SHUFFLE, len(test_dataloader.dataset)))
    # config_logging(os.path.join(cfg.CHECKPOINT_PATH, 'logging.txt'), cfg, access_mode='w+')

    # Final activation
    activation_func = create_activation(cfg.MODEL.ACTIVATION)

    # TODO: device change to captial
    eval_instance = Evaluator(cfg,
                              model,
                              test_dataloader,
                              logger=LOGGER,
                              device=cfg.device,
                              activation_func=activation_func,
                              USE_TENSORBOARD=True,
                              USE_CUDA=True)

    eval_instance.evaluate()


if __name__ == '__main__':
    # main(CONFIG_PATH)

    
    import matplotlib.pyplot as plt
    from data.data_utils import get_files
    root = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\crop\32x64x64-10\positive\Image'
    # root = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-preprocess\crop\32x64x64-10\positive\Image'
    f_list = get_files(root, 'npy')

    for idx, f in enumerate(f_list):
        # if idx != 97: continue
        print(idx)
        # f = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\crop\32x64x64-10\positive\Image'
        # f = os.path.join(f, rf'0011-TMH0011.npy')
        mf = f.replace('Image', 'Mask')
        x = np.load(f)
        y = np.load(mf)
        for x_, y_ in zip(x, y):
            if np.sum(y_):
                plt.imshow(x_, 'gray')
                plt.imshow(y_, alpha=0.2)
                plt.show()


