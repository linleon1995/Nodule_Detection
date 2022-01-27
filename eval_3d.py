import site_path
import os
from ResNet_3d import build_3d_resnet
from torch.utils.data import Dataset, DataLoader
from dataloader import Luna16CropDataset
import numpy as np
import random
import torch
from pprint import pprint
import tensorboardX

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
        print('Precision', metrics.precision(np.sum(self.eval_tool.total_tp), np.sum(self.eval_tool.total_fp)))
        print('Recall', metrics.recall(np.sum(self.eval_tool.total_tp), np.sum(self.eval_tool.total_fn)))
        print('Acc', self.avg_test_acc)
    

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

    test_dataset = Luna16CropDataset(config.DATA.DATA_PATH, config.DATA.CROP_RANGE, mode='test')
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

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
    main(CONFIG_PATH)