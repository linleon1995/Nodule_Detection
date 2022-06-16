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
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import time
from nodule_classification.model.MatchingNet import MatchingNetwork_3d
from nodule_classification.data.matchnet_utils import build_support_set, matchingnet_trainer

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
                save_path,
                activation_func=None,
                USE_TENSORBOARD=True,
                USE_CUDA=True,):
        self.cfg = cfg
        self.model = model
        self.test_dataloader = test_dataloader
        self.logger = logger
        self.iterations = 0
        self.device = device
        self.activation_func = activation_func
        self.USE_TENSORBOARD = USE_TENSORBOARD
        self.save_path = save_path
        # self.checkpoint_path = self.cfg.EVAL.CHECKPOINT_PATH
        
    def evaluate(self):
        self.model.eval()
        # self.eval_tool = metrics.SegmentationMetrics(self.cfg.MODEL.NUM_CLASSES, ['accuracy'])
        valid_samples = len(self.test_dataloader.dataset)
        total_labels, total_preds = [], []
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
            # evals = self.eval_tool(labels, prediction)
            total_labels.append(labels)
            total_preds.append(prediction)

            input_var = input_var.cpu().detach().numpy()
            start = time.time()
            vis_for_img(
                prob, labels, input_var[0,0], 
                os.path.join(self.save_path, f'{data["tmh_name"][0]}', str(idx))
            )
            end = time.time()
            print(f'{idx+1}/{valid_samples} {data["tmh_name"][0]} {end-start} sec')

        # self.avg_test_acc = metrics.accuracy(
                # np.sum(self.eval_tool.total_tp), np.sum(self.eval_tool.total_fp), np.sum(self.eval_tool.total_fn), np.sum(self.eval_tool.total_tn)).item()
        total_labels = np.concatenate(total_labels)
        total_preds = np.concatenate(total_preds)
        result = metrics.cls_metrics(total_labels, total_preds)
        return result

class MatchingNet_Evaluator(Evaluator):
    def __init__(self, 
                cfg,
                model,
                test_dataloader,
                logger,
                device,
                save_path,
                activation_func=None,
                USE_TENSORBOARD=True,
                USE_CUDA=True,):
        super().__init__(cfg,
                        model,
                        test_dataloader,
                        logger,
                        device,
                        save_path,
                        activation_func=None,
                        USE_TENSORBOARD=True,
                        USE_CUDA=True)
    def evaluate(self, support_set_x, support_set_y):
        self.model.eval()
        # self.eval_tool = metrics.SegmentationMetrics(self.cfg.MODEL.NUM_CLASSES, ['accuracy'])
        valid_samples = len(self.test_dataloader.dataset)
        total_labels, total_preds = [], []
        support_set_x = torch.from_numpy(support_set_x).to(self.device)
        support_set_y = torch.from_numpy(support_set_y).to(self.device)
        for idx, data in enumerate(self.test_dataloader):
            input_var, labels = data['input'], data['target']
            labels = labels.long()
            input_var, labels = input_var.to(self.device), labels.to(self.device)
            # outputs = self.model(input_var)

            
            support_set_images = torch.tile(
                torch.unsqueeze(support_set_x, dim=1), (1, 3, 1, 1, 1))
            support_set_y_one_hot = torch.unsqueeze(support_set_y, dim=1)

            support_set_images = support_set_images.to(torch.float)
            accuracy, crossentropy_loss, prob, pred = self.model(
                support_set_images, support_set_y_one_hot, input_var, labels)
                
            # if self.activation_func:
            #     if self.cfg.MODEL.ACTIVATION == 'softmax':
            #         prob = self.activation_func(outputs)
            #     else:
            #         prob = self.activation_func(outputs)
            # else:
            #     prob = outputs
            prediction = torch.argmax(prob, dim=1)
            labels = labels[:,0]

            labels = labels.cpu().detach().numpy()
            prediction = prediction.cpu().detach().numpy()
            # evals = self.eval_tool(labels, prediction)
            total_labels.append(labels)
            total_preds.append(prediction)

            input_var = input_var.cpu().detach().numpy()
            start = time.time()
            vis_for_img(
                prob, labels, input_var[0,0], 
                os.path.join(self.save_path, f'{data["tmh_name"][0]}', str(idx))
            )
            end = time.time()
            print(f'{idx+1}/{valid_samples} {data["tmh_name"][0]} {end-start} sec')

        # self.avg_test_acc = metrics.accuracy(
                # np.sum(self.eval_tool.total_tp), np.sum(self.eval_tool.total_fp), np.sum(self.eval_tool.total_fn), np.sum(self.eval_tool.total_tn)).item()
        total_labels = np.concatenate(total_labels)
        total_preds = np.concatenate(total_preds)
        result = metrics.cls_metrics(total_labels, total_preds)
        return result
        

def vis_for_img(prob, labels, input_var, save_path):
    os.makedirs(save_path, exist_ok=True)
    fig, ax = plt.subplots(1,1)
    for i in range(input_var.shape[0]):
        # print(i)
        ax.imshow(input_var[i], 'gray')
        ax.set_title(f'class 1 prob {prob[0,1]}  label {labels[0]}')
        fig.savefig(os.path.join(save_path, f'{i}.png'))
        ax.cla()
    plt.close()


def main(config_reference):
    # Configuration
    cfg = load_config(config_reference)
    cfg = DictAsMember(cfg)
    device = get_device(cfg.get('device', None))
    cfg.device = device
    pprint(cfg)

    # TODO: the fold info in here has some prolem
    coco_list = build_coco_path(cfg.DATA.COCO_PATH[cfg.DATA.NAME], cfg.CV.FOLD, cfg.CV.ASSIGN, mode='eval')
    cv_precision, cv_recall, cv_specificity, cv_accuracy = [], [], [], []
    for (fold, test_coco) in coco_list:
        checkpoint_path = os.path.join(cfg.EVAL.CHECKPOINT_ROOT, str(fold), cfg.EVAL.CHECKPOINT)
        # print(50*'-', f'Fold {fold}', 50*'-')
        fold_result = eval(cfg, test_coco, checkpoint_path)  
        mean_precision, mean_recall, mean_specificity, accuracy, cm = fold_result
        cv_precision.append(mean_precision)
        cv_recall.append(mean_recall)
        cv_specificity.append(mean_specificity)
        cv_accuracy.append(accuracy)
    mean_cv_precision = np.mean(cv_precision)
    mean_cv_recall = np.mean(cv_recall)
    mean_cv_specificity = np.mean(cv_specificity)
    mean_cv_accuracy = np.mean(cv_accuracy)

    print(50*'-', f'CV', 50*'-')
    print(f'mean cv precision {mean_cv_precision}')
    print(f'mean cv recall {mean_cv_recall}')
    print(f'mean cv specificity {mean_cv_specificity}')
    print(f'mean cv accuracy {mean_cv_accuracy}')


def eval(cfg, test_coco, checkpoint_path):
    if cfg.MODEL.NAME == 'MatchingNet':
        model = MatchingNetwork_3d(keep_prob=0.9, 
                                   batch_size=cfg.DATA.BATCH_SIZE,
                                   num_channels=3, 
                                   learning_rate=None, 
                                   fce=False,
                                   use_cuda=True,
                                   model_depth=cfg.MODEL.DEPTH,
                                   n_classes=cfg.MODEL.NUM_CLASSES,)
        support_set_x, support_set_y = build_support_set(
            n=32, n_class=cfg.MODEL.NUM_CLASSES)
    elif cfg.MODEL.NAME == '3dResnet':
        model = build_3d_resnet(model_depth=cfg.MODEL.DEPTH, n_classes=cfg.MODEL.NUM_CLASSES, conv1_t_size=7, conv1_t_stride=2)
    
    state_key = torch.load(checkpoint_path, map_location=cfg.device)
    model.load_state_dict(state_key['net'])
    model = model.to(cfg.device)

    test_seriesuid = get_pids_from_coco(test_coco)
    if cfg.DATA.TASK == 'Nodule':
        test_dataset = BaseNoduleClsDataset(cfg.DATA.DATA_PATH[cfg.DATA.NAME], cfg.DATA.CROP_RANGE, 
                                        test_seriesuid, data_augmentation=False)
    elif cfg.DATA.TASK == 'Malignancy':
        test_dataset = BaseMalignancyClsDataset(cfg.DATA.DATA_PATH[cfg.DATA.NAME], cfg.DATA.CROP_RANGE, 
                                            test_seriesuid, data_augmentation=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0)

    # Logger
    LOGGER.info("Start Evaluation!!")
    LOGGER.info("Batch size: {} Shuffling Data: {} Testing Samples: {}".
            format(cfg.DATA.BATCH_SIZE, cfg.DATA.SHUFFLE, len(test_dataloader.dataset)))
    # config_logging(os.path.join(cfg.EVAL.CHECKPOINT_PATH, 'logging.txt'), cfg, access_mode='w+')

    # Final activation
    activation_func = create_activation(cfg.MODEL.ACTIVATION)

    if cfg.MODEL.NAME == 'MatchingNet':
        eval_instance = MatchingNet_Evaluator(cfg,
                                model,
                                test_dataloader,
                                logger=LOGGER,
                                device=cfg.device,
                                save_path=os.path.join(cfg.EVAL.CHECKPOINT_ROOT, 'eval'),
                                activation_func=activation_func,
                                USE_TENSORBOARD=True,
                                USE_CUDA=True)
        result = eval_instance.evaluate(support_set_x, support_set_y)
    else:
        eval_instance = Evaluator(cfg,
                                model,
                                test_dataloader,
                                logger=LOGGER,
                                device=cfg.device,
                                save_path=os.path.join(cfg.EVAL.CHECKPOINT_ROOT, 'eval'),
                                activation_func=activation_func,
                                USE_TENSORBOARD=True,
                                USE_CUDA=True)
        result = eval_instance.evaluate()
    return result


if __name__ == '__main__':
    start_time = time.time()
    main(CONFIG_PATH)
    end_time = time.time()
    print(end_time-start_time, (end_time-start_time)/60)

    