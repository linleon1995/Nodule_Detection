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
from nodule_classification.data.data_loader import BaseNoduleClsDataset, BaseMalignancyClsDataset
from nodule_classification.data.build_crop_dataset import build_coco_path
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
        # self.checkpoint_path = self.cfg.EVAL.CHECKPOINT_PATH
        
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

    coco_list = build_coco_path(cfg.DATA.COCO_PATH[cfg.DATA.NAME], cfg.CV.FOLD, cfg.CV.ASSIGN, mode='eval')
    for fold, test_coco in enumerate(coco_list):
        checkpoint_path = os.path.join(cfg.EVAL.CHECKPOINT_ROOT, str(fold), cfg.EVAL.CHECKPOINT)
        eval(cfg, test_coco, checkpoint_path)  


def eval(cfg, test_coco, checkpoint_path):
    model = build_3d_resnet(model_depth=cfg.MODEL.DEPTH, n_classes=cfg.MODEL.NUM_CLASSES, conv1_t_size=7, conv1_t_stride=2)
    state_key = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state_key['net'])
    model = model.to(cfg.device)

    test_seriesuid = get_pids_from_coco(test_coco)
    test_dataset = BaseMalignancyClsDataset(cfg.DATA.DATA_PATH[cfg.DATA.NAME], cfg.DATA.CROP_RANGE, 
                                            test_seriesuid, data_augmentation=False)
    # test_dataset = BaseNoduleClsDataset(cfg.DATA.DATA_PATH[cfg.DATA.NAME], cfg.DATA.CROP_RANGE, 
    #                                   test_seriesuid, data_augmentation=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    # Logger
    LOGGER.info("Start Evaluation!!")
    LOGGER.info("Batch size: {} Shuffling Data: {} Testing Samples: {}".
            format(cfg.DATA.BATCH_SIZE, cfg.DATA.SHUFFLE, len(test_dataloader.dataset)))
    # config_logging(os.path.join(cfg.EVAL.CHECKPOINT_PATH, 'logging.txt'), cfg, access_mode='w+')

    # Final activation
    activation_func = create_activation(cfg.MODEL.ACTIVATION)

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
    main(CONFIG_PATH)


