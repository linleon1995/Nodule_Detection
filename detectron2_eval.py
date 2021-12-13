from detectron2.evaluation import COCOEvaluator, inference_on_dataset, SemSegEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

from utils import cv2_imshow
from convert_to_coco_structure import lidc_to_datacatlog_valid
import logging
from sklearn.metrics import confusion_matrix
logging.basicConfig(level=logging.INFO)

import site_path
from modules.utils import metrics
from modules.utils import metrics2
from modules.utils import evaluator


def eval(cfg):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    predictor = DefaultPredictor(cfg)

    register_coco_instances("my_dataset_valid", {}, "annotations_valid.json", rf"C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Image")
    # DatasetCatalog.register("my_dataset_valid", lidc_to_datacatlog_valid)
    dataset_dicts = DatasetCatalog.get("my_dataset_valid")
    metadata = MetadataCatalog.get("my_dataset_valid")

    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=metadata, 
                    scale=1.0, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2_imshow(out.get_image()[:, :, ::-1])


    evaluator = COCOEvaluator("my_dataset_valid", cfg, distributed=False, output_dir="./output")
    # evaluator = SemSegEvaluator("my_dataset_valid", distributed=False, num_classes=2, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, "my_dataset_valid")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    # another equivalent way to evaluate the model is to use `trainer.test`


def compute_mean_dsc(ref=None, test=None, **metrics_kwargs):
    """Compute the mean intersection-over-union via the confusion matrix."""
    total_cm = metrics_kwargs.pop("total_cm")
    sum_over_row = np.sum(total_cm, axis=0).astype(float)
    sum_over_col = np.sum(total_cm, axis=1).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    denominator = sum_over_row + sum_over_col

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = np.sum((denominator != 0).astype(float))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))

    dscs = 2*cm_diag / denominator

    print('Dice Score Simililarity for each class:')
    for i, dsc in enumerate(dscs):
        print('    class {}: {:.4f}'.format(i, dsc))

    # If the number of valid entries is 0 (no classes) we return 0.
    m_dsc = np.where(
        num_valid_entries > 0,
        np.sum(dscs) / num_valid_entries,
        0)
    print('mean Dice Score Simililarity: {:.4f}'.format(float(m_dsc)))
    return m_dsc, dscs


def compute_mean_iou(ref=None, test=None, **metrics_kwargs):
    """Compute the mean intersection-over-union via the confusion matrix."""
    total_cm = metrics_kwargs.pop("total_cm")
    sum_over_row = np.sum(total_cm, axis=0).astype(float)
    sum_over_col = np.sum(total_cm, axis=1).astype(float)
    cm_diag = np.diagonal(total_cm).astype(float)
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = np.sum((denominator != 0).astype(float))

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = np.where(
        denominator > 0,
        denominator,
        np.ones_like(denominator))

    ious = cm_diag / denominator

    print('Intersection over Union for each class:')
    for i, iou in enumerate(ious):
        print('    class {}: {:.4f}'.format(i, iou))

    # If the number of valid entries is 0 (no classes) we return 0.
    m_iou = np.where(
        num_valid_entries > 0,
        np.sum(ious) / num_valid_entries,
        0)
    print('mean Intersection over Union: {:.4f}'.format(float(m_iou)))
    return m_iou


def eval_test(gt, pred, num_class):
    # gt = np.eye(num_class)[gt]

    gt = np.reshape(gt, [-1])
    # pred = np.eye(num_class)[pred]
    pred = np.reshape(pred, [-1])
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(gt, pred, labels=np.arange(0, num_class))
    tp1, fp1, fn1 = cm[0,0], cm[0,1]+cm[0,2], cm[1,0]+cm[2,0]
    tp2, fp2, fn2 = cm[1,1], cm[1,0]+cm[1,2], cm[0,1]+cm[2,1]
    tp3, fp3, fn3 = cm[2,2], cm[2,0]+cm[2,1], cm[0,2]+cm[1,2]

    if tp1+fp1+fn1==0:
        dsc1, iou1 = -1, -1
    else:
        dsc1, iou1 = 2*tp1/(2*tp1+fp1+fn1), tp1/(tp1+fp1+fn1)

    if tp2+fp2+fn2==0:
        dsc2, iou2 = -1, -1
    else:
        dsc2, iou2 = 2*tp2/(2*tp2+fp2+fn2), tp2/(tp2+fp2+fn2)

    if tp3+fp3+fn3==0:
        dsc3, iou3 = -1, -1
    else:
        dsc3, iou3 = 2*tp3/(2*tp3+fp3+fn3), tp3/(tp3+fp3+fn3)
    return [dsc1, dsc2, dsc3], [iou1, iou2, iou3]


def save_mask(cfg):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    predictor = DefaultPredictor(cfg)

    register_coco_instances("my_dataset_valid", {}, "annotations_valid.json", rf"C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Image")
    # DatasetCatalog.register("my_dataset_valid", lidc_to_datacatlog_valid)
    dataset_dicts = DatasetCatalog.get("my_dataset_valid")
    metadata = MetadataCatalog.get("my_dataset_valid")

    # evaluator = metrics.SegmentationMetrics(num_class=cfg.MODEL.ROI_HEADS.NUM_CLASSES)
    total_iou, total_dsc = [], []
    total_cm = 0
    lidc_evaluator = evaluator.ClassificationEvaluator(num_class=cfg.MODEL.ROI_HEADS.NUM_CLASSES+1)
    lidc_evaluator.register_new_metrics({"DSC": metrics2.mean_dsc, 'IoU': metrics2.mean_iou})
    for idx, d in enumerate(dataset_dicts):

        img_file_name = d["file_name"]
        mask_file_name = d['image_id'].replace('NI', 'MA')
        img = cv2.imread(img_file_name)
        outputs = predictor(img)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        # v = Visualizer(img[:, :, ::-1],
        #             metadata=metadata, 
        #             scale=1.0, 
        #             instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        # )
        pred_mask = outputs["instances"]._fields['pred_masks'].cpu().detach().numpy() 
        pred_classes = outputs["instances"]._fields['pred_classes'].cpu().detach().numpy() 
        pred_classes = np.reshape(pred_classes, (pred_classes.shape[0], 1, 1))
        pred_classes += 1
        # pred_mask = np.sum(pred_mask, axis=0)
        pred_mask = np.sum(pred_mask*pred_classes, axis=0)
        # print(pred_classes.shape[0])
        # if pred_classes.shape[0] > 0:
        #     print(3)
        # if np.sum(pred_classes==2) > 0 and np.sum(pred_classes==1) > 0:
            # print(pred_classes)

        # plt.imshow(pred_mask, vmin=0, vmax=2)
        # plt.savefig(f'vis/show/{mask_file_name}.png')
        

        # Get ground truth
        gt_file_name = img_file_name.replace('NI', 'MA').replace('Image', 'Semantic_Mask')
        gt_mask = cv2.imread(gt_file_name)
        # cv2_imshow((gt_mask+np.uint8(np.tile(pred_mask[...,np.newaxis], [1,1,3])))*63)

        # Evaluation
        # if idx == 42:
        #     print(3)
        gt_mask = np.int32(gt_mask[...,0])
        pred_mask = np.int32(pred_mask)
        # gt_mask = np.reshape(gt_mask, [-1])
        # pred_mask = np.reshape(pred_mask, [-1])
        # cm = confusion_matrix(np.reshape(gt_mask, [-1]), np.reshape(pred_mask, [-1]), labels=np.arange(0, cfg.MODEL.ROI_HEADS.NUM_CLASSES+1))
        lidc_evaluator.evaluate(np.reshape(gt_mask, [-1]), np.reshape(pred_mask, [-1]))
        # total_cm += cm
        # iou, dsc = eval_test(gt_mask, pred_mask, num_class=cfg.MODEL.ROI_HEADS.NUM_CLASSES+1)
        # print(f'IoU: {iou} DSC: {dsc}')
        # mean_iou = np.mean([x for x in iou if x > 0])
        # mean_dsc = np.mean([x for x in dsc if x > 0])
        # total_iou.append(mean_iou) 
        # total_dsc.append(mean_dsc)
        # if idx > 10: break

        # # Save figure
        # fig, ax = plt.subplots(1,3)
        # # if np.sum(gt_mask==1):
        # # num_class = 3
        # ax[0].imshow(gt_mask, vmin=0, vmax=cfg.MODEL.ROI_HEADS.NUM_CLASSES-1)
        # ax[0].set_title('gt')
        # ax[1].imshow(pred_mask, vmin=0, vmax=cfg.MODEL.ROI_HEADS.NUM_CLASSES-1)
        # ax[1].set_title('pred')
        # ax[2].imshow(gt_mask, vmin=0, vmax=cfg.MODEL.ROI_HEADS.NUM_CLASSES-1)
        # ax[2].imshow(pred_mask, alpha=0.2, vmin=0, vmax=cfg.MODEL.ROI_HEADS.NUM_CLASSES-1)
        # ax[2].set_title('compare')
        # plt.savefig(f'vis/show2/{mask_file_name}.png')
        # plt.close(fig)

        
        

        # Save images
        # cv2_imshow(np.uint8(np.tile(pred_mask[...,np.newaxis], [1,1,3])*255))
        # cv2_imshow(np.uint8(gt_mask*255))
        # cv2_imshow((np.uint8(gt_mask[...,0]*255)+np.uint8(pred_mask*255))//2)

        # pred_mask = np.uint8(np.tile(pred_mask[...,np.newaxis], [1,1,3])*127)
        print(idx, mask_file_name)
        # cv2.imwrite(f'vis/mask2/{mask_file_name}.png', pred_mask)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2.imwrite(f'vis/instance/{mask_file_name}.png', out.get_image()[:, :, ::-1])
    # print(3)
    total_aggregation = lidc_evaluator.get_aggregation(np.mean)
    mean_iou, mean_dsc = total_aggregation['IoU'], total_aggregation['DSC']
    print(f'mean IoU: {mean_iou} mean DSC: {mean_dsc}')


if __name__ == '__main__':
    check_point_path = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output\run_001'
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_final.pth")  # path to the model we just trained
    # cfg.MODEL.WEIGHTS = os.path.join(check_point_path, "model_0004999.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.OUTPUT_DIR = check_point_path
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    # eval(cfg)
    save_mask(cfg)