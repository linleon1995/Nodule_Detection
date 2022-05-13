import os
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

from utils import configuration
from utils.configuration import DictAsMember


def get_eval_config():
    cfg = {}
    yaml_cfg = configuration.load_config(f'config_file/test.yml', dict_as_member=True)
    common_cfg = common_config()
    cfg.update(yaml_cfg)
    cfg.update(common_cfg)
    
    if yaml_cfg.MODEL.backend == 'd2':
        checkpoint_path = os.path.join(yaml_cfg.MODEL.OUTPUT_DIR, yaml_cfg.MODEL.WEIGHTS)
        d2_cfg = d2_eval_config(checkpoint_path)
        cfg['d2'] = d2_cfg

    cfg = DictAsMember(cfg)
    set_save_path(cfg)
    return cfg


def d2_eval_config(checkpoint_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATALOADER.NUM_WORKERS = 0
    
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3  # set a custom testing threshold
    
    # cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
    cfg.INPUT.MASK_FORMAT = 'bitmask'
    cfg.INPUT.MIN_SIZE_TEST = 1120
    cfg.MODEL.WEIGHTS = checkpoint_path

    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[4,  8,  16,  32,  64]]
    # cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.2]]
    # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8,  16,  32,  64, 128]]
    # cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 20
    # cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 20
    # NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
    return cfg


def common_config():
    cfg = DictAsMember({})
    cfg.DATA_SPLIT = 'test'

    # False Positive reduction
    cfg.NODULE_CLS_PROB = 0.75
    cfg.nodule_cls = True
    cfg.crop_range = [48, 48, 48]
    cfg.FP_reducer_checkpoint = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\checkpoints\run_028\ckpt_best.pth'
    cfg.lung_mask_filtering = True
    cfg.remove_1_slice = True
    cfg.remove_unusual_nodule_by_lung_size = True
    cfg.remove_unusual_nodule_by_ratio = False
    cfg.lung_size_threshold = 0.4
    cfg.pred_slice_threshold = 1
    cfg.MATCHING_THRESHOLD = 0.1

    # Exepriment
    cfg.MAX_SAVE_IMAGE_CASES = 2
    cfg.MAX_TEST_CASES = None
    # cfg.ONLY_NODULES = True
    cfg.SAVE_ALL_COMPARES = True
    cfg.TEST_BATCH_SIZE = 2
    cfg.SAVE_ALL_IMAGES = False


    cfg.connectivity = 26
    cfg.area_threshold = 0
    return cfg


def set_save_path(cfg):
    # Get directory name
    run = os.path.split(cfg.MODEL.OUTPUT_DIR)[1]
    weight = cfg.MODEL.WEIGHTS.split('.')[0]
    # cfg.SAVE_PATH = rf'C:\Users\test\Desktop\Leon\Weekly\1227'
    dir_name = ['maskrcnn', f'{run}', f'{weight}']
    if 'd2' in cfg:
        dir_name.append(f'{cfg.d2.MODEL.ROI_HEADS.SCORE_THRESH_TEST}')
    dir_name.insert(0, '1SR') if cfg.remove_1_slice else dir_name
    dir_name.insert(0, 'RUNR') if cfg.remove_unusual_nodule_by_ratio else dir_name
    dir_name.insert(0, f'RUNLS_TH{cfg.lung_size_threshold}') if cfg.remove_unusual_nodule_by_lung_size else dir_name
    dir_name.insert(0, 'LMF') if cfg.lung_mask_filtering else dir_name
    FPR_model_code = os.path.split(os.path.split(cfg.FP_reducer_checkpoint)[0])[1]
    dir_name.insert(0, f'NC#{FPR_model_code}') if cfg.nodule_cls else dir_name
    if 'd2' in cfg:
        dir_name.insert(0, str(cfg.d2.INPUT.MIN_SIZE_TEST))
    cfg.SAVE_PATH = os.path.join(cfg.MODEL.OUTPUT_DIR, '-'.join(dir_name))

    
# def get_model_path():
#     checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_001'
#     checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_003'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_004'
#     checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_005'
#     checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_006'
#     checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_007'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_010'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_016'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_017'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_018'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_019'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_020'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_021'
#     checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_022'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_023'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_024'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_026'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_027'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_028'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_032'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_033'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_034'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_035'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_036'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_037'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_040'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_041'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_044'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_045'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_046'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_048'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_049'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_051'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_053'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_052'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_055'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_056'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_057'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_058'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_059'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_060'
#     # checkpoint_path = rf'C:\Users\test\Desktop\Leon\Projects\Nodule_Detection\output\run_061'
#     output_dir = checkpoint_path

#     model_weight = os.path.join(checkpoint_path, "model_0005999.pth")  # path to the model we just trained
#     # model_weight = os.path.join(checkpoint_path, "model_0005999.pth")  # path to the model we just trained
#     # model_weight = os.path.join(checkpoint_path, "model_final.pth")  # path to the model we just trained


#     output_dir = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\pretrained_weights\Unet3D-genesis_chest_ct\run_004'
#     model_weight = os.path.join(output_dir, "ckpt-best.pt")  # path to the model we just trained
    

#     # output_dir = rf'C:\Users\test\Desktop\Leon\Projects\ModelsGenesis\keras\downstream_tasks\models\ncs\run_4'
#     # model_weight = os.path.join(output_dir, "Vnet-genesis.h5")
#     return output_dir, model_weight