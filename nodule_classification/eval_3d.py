import site_path
import os
from model.ResNet_3d import build_3d_resnet
from torch.utils.data import Dataset, DataLoader
from data.dataloader import Luna16CropDataset
from data.dataloader import ASUSCropDataset
import numpy as np
import random
import torch
from pprint import pprint
import tensorboardX
from data.luna16_data_preprocess import LUNA16_CropRange_Builder
from data.asus_crop_preprocess import ASUS_CropRange_Builder
from utils.utils import build_size_figure

from modules.train import trainer
from modules.utils import configuration
from modules.utils import train_utils
from modules.utils import metrics

CONFIG_PATH = 'config/test_config.yml'
LOGGER = train_utils.get_logger('test')


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
                 config,
                model,
                test_dataloader,
                logger,
                device,
                activation_func=None,
                USE_TENSORBOARD=True,
                USE_CUDA=True):
        self.config = config
        self.model = model
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.iterations = 0
        self.device = device
        self.activation_func = activation_func
        self.USE_TENSORBOARD = USE_TENSORBOARD
        self.checkpoint_path = self.config.CHECKPOINT_PATH
        
    def evaluate(self):
        self.model.eval()
        self.eval_tool = metrics.SegmentationMetrics(self.config.MODEL.NUM_CLASSES, ['accuracy'])
        valid_samples = len(self.test_dataloader.dataset)
        for idx, data in enumerate(self.test_dataloader):
            input_var, labels = data['input'], data['target']
            labels = labels.long()
            input_var, labels = input_var.to(self.device), labels.to(self.device)
            outputs = self.model(input_var)
            
            if self.activation_func:
                if self.config.MODEL.ACTIVATION == 'softmax':
                    prob = self.activation_func(outputs, dim=1)
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
        print(f'Precision: {precision*100:.02f}')
        print(f'Recall: {recall*100:.02f}')
        print('Acc', self.avg_test_acc)
        print(f'TP: {self.eval_tool.total_tp}', 
              f'FP: {self.eval_tool.total_fp}',
              f'TN: {self.eval_tool.total_tn}',
              f'FN: {self.eval_tool.total_fn}')
    

def main(config_reference):
    # Configuration
    config = configuration.load_config(config_reference)
    config = train_utils.DictAsMember(config)
    device = configuration.get_device(config.get('device', None))
    config.device = device
    pprint(config)

    model = build_3d_resnet(model_depth=config.MODEL.DEPTH, n_classes=config.MODEL.NUM_CLASSES, conv1_t_size=7, conv1_t_stride=2)
    state_key = torch.load(config.CHECKPOINT_PATH, map_location=config.device)
    model.load_state_dict(state_key['net'])
    model = model.to(config.device)

    test_datasets = []
    for dataset_name in config.DATA.NAME:
        
        if dataset_name == 'LUNA16':
            file_name_key = LUNA16_CropRange_Builder.get_filename_key(config.DATA.CROP_RANGE, config.DATA.NPratio)
            data_path = os.path.join(config.DATA.DATA_PATH[dataset_name], file_name_key)
            test_dataset = Luna16CropDataset(data_path, config.DATA.CROP_RANGE, mode='test')
        elif dataset_name in ['ASUS-B', 'ASUS-M']:
            file_name_key = ASUS_CropRange_Builder.get_filename_key(config.DATA.CROP_RANGE, config.DATA.NPratio)
            data_path = os.path.join(config.DATA.DATA_PATH[dataset_name], file_name_key)
            test_dataset = ASUSCropDataset(data_path, config.DATA.CROP_RANGE, negative_to_positive_ratio=config.DATA.NPratio_test, nodule_type=dataset_name, mode='test')

        print(f'Dataset: {config.DATA.NAME} Test number: {len(test_dataset)}')
        test_datasets.append(test_dataset)

    total_test_dataset = torch.utils.data.ConcatDataset(test_datasets)
    test_dataloader = DataLoader(total_test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)


    # # test_dataset = Luna16CropDataset(config.DATA.DATA_PATH, config.DATA.CROP_RANGE, mode='test')
    # # test_dataset = ASUSCropDataset(config.DATA.DATA_PATH, config.DATA.CROP_RANGE, nodule_type='ASUS-B', mode='test')
    # test_dataset = ASUSCropDataset(config.DATA.DATA_PATH, config.DATA.CROP_RANGE, nodule_type='ASUS-M', mode='test')
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    # Logger
    LOGGER.info("Start Evaluation!!")
    LOGGER.info("Batch size: {} Shuffling Data: {} Testing Samples: {}".
            format(config.DATA.BATCH_SIZE, config.DATA.SHUFFLE, len(test_dataloader.dataset)))
    # train_utils.config_logging(os.path.join(config.CHECKPOINT_PATH, 'logging.txt'), config, access_mode='w+')

    # Final activation
    activation_func = train_utils.create_activation(config.MODEL.ACTIVATION)

    # TODO: device change to captial
    eval_instance = Evaluator(config,
                                model,
                                test_dataloader,
                                logger=LOGGER,
                                device=config.device,
                                activation_func=activation_func,
                                USE_TENSORBOARD=True,
                                USE_CUDA=True)

    eval_instance.evaluate()


if __name__ == '__main__':
    # main(CONFIG_PATH)

    # min_size, max_size, size_step = 0, 20001, 20000//(5-1)
    # size_threshold = np.arange(min_size, max_size, size_step)
    # print(size_threshold)

    data = [{'size': 500, 'score': 0.67}, {'size': 5000, 'score': 0.77}, {'size': 50000, 'score': 0.87}]
    build_size_figure(data)