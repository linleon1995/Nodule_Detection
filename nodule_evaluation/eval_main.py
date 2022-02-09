import os
from re import L
import numpy as np
from eval_utils import volumetric_data_eval


def evaluation(data_path, save_path):
    vol_metric = volumetric_data_eval(save_path)
    target_files = [path for path in os.listdir(data_path) if os.path.split(path)[1].startswith('target')]
    pred_files = [path for path in os.listdir(data_path) if os.path.split(path)[1].startswith('pred')]

    for target_path, pred_path in zip(target_files, pred_files):
        # Get data pair (target volume, predict volume)
        target, pred = np.load(os.path.join(data_path, target_path)), np.load(os.path.join(data_path, pred_path))

        # Get volume evaluation
        vol_nodule_infos = vol_metric.calculate(target, pred)
        print(vol_nodule_infos)

    # Get total evaluation
    nodule_tp, nodule_fp, nodule_fn, nodule_precision, nodule_recall = vol_metric.evaluation(show_evaluation=True)
    

def main():
    data_path = rf'example_data/'
    save_path = rf'eval/'
    evaluation(data_path, save_path)

if __name__ == '__main__':
    main()
    
    
    