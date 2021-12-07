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

from utils import cv2_imshow
from convert_to_coco_structure import lidc_to_datacatlog_valid
import logging
logging.basicConfig(level=logging.INFO)


def eval(cfg):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    predictor = DefaultPredictor(cfg)

    register_coco_instances("my_dataset_valid", {}, "annotations_valid.json", rf"C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Image")
    # DatasetCatalog.register("my_dataset_valid", lidc_to_datacatlog_valid)
    dataset_dicts = DatasetCatalog.get("my_dataset_valid")
    # MetadataCatalog.get("my_dataset_valid").set(stuff_classes=["benign", "malignant"])
    # MetadataCatalog.get("my_dataset_valid").class_names = ["benign", "malignant"]
    for d in random.sample(dataset_dicts, 3):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get("my_dataset_valid"), 
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


def save_mask(cfg):
    # Inference should use the config with parameters that are used in training
    # cfg now already contains everything we've set previously. We changed it a little bit for inference:
    predictor = DefaultPredictor(cfg)

    register_coco_instances("my_dataset_valid", {}, "annotations_valid.json", rf"C:\Users\test\Desktop\Leon\Datasets\LIDC-IDRI-process\LIDC-IDRI-Preprocessing-png\Image")
    # DatasetCatalog.register("my_dataset_valid", lidc_to_datacatlog_valid)
    dataset_dicts = DatasetCatalog.get("my_dataset_valid")
    # MetadataCatalog.get("my_dataset_valid").set(stuff_classes=["benign", "malignant"])
    # MetadataCatalog.get("my_dataset_valid").class_names = ["benign", "malignant"]
    for idx, d in enumerate(dataset_dicts):    
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get("my_dataset_valid"), 
                    scale=1.0, 
                    instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
        )
        pred_mask = outputs["instances"]._fields['pred_masks'].cpu().detach().numpy() 
        pred_classes = outputs["instances"]._fields['pred_classes'].cpu().detach().numpy() 
        pred_classes
        pred_mask = np.sum(pred_mask, axis=0)
        pred_mask = np.uint8(np.tile(pred_mask[...,np.newaxis], [1,1,3])*255)
        file_name = d['image_id'].replace('NI', 'MA')
        cv2.imwrite(f'mask/{file_name}.png', pred_mask)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])


if __name__ == '__main__':
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    # cfg.DATASETS.TRAIN = ("my_dataset_train",)
    # cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 0
    # cfg.SOLVER.IMS_PER_BATCH = 8
    # cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    # cfg.SOLVER.MAX_ITER = 300   # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    # cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.OUTPUT_DIR = rf'C:\Users\test\Desktop\Leon\Projects\detectron2\output'
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

    eval(cfg)
    # save_mask(cfg)