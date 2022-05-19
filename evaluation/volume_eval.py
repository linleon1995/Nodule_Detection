import os
import numpy as np
import matplotlib.pyplot as plt
import time

# TODO: delte these two globals
CONNECTIVITY = 26
AREA_THRESHOLD = 20



class volumetric_data_eval():
    def __init__(self, model_name, save_path, dataset_name, match_threshold=0.5, max_nodule_num=1):
        self.model_name = model_name
        self.save_path = save_path
        self.dataset_name = dataset_name
        self.match_threshold = match_threshold
        self.max_nodule_num = max_nodule_num
        self.PixelTP, self.PixelFP, self.PixelFN = [], [] ,[]
        self.VoxelTP, self.VoxelFP, self.VoxelFN = [], [] ,[]
        self.DoubleDetections = []
        self.num_nodule = 0
        self.num_case = 0
        os.makedirs(self.save_path, exist_ok=True)
        self.eval_file = open(os.path.join(self.save_path, 'evaluation.txt'), 'w+')
        self.mean_iou, self.mean_dsc = None, None

    def calculate(self, target_study, pred_study):
        self.num_case += 1
        self.num_nodule += len(target_study.nodule_instances)
        assert np.shape(target_study.category_volume) == np.shape(pred_study.category_volume)
        tp, fp, fn, doubleCandidatesIgnored = self._3D_evaluation_2(target_study, pred_study)
        # self._2D_evaluation(target_study, pred_study)
        return tp, fp, fn, doubleCandidatesIgnored
    
    def test(self, labels, y_pred):
        true_objects = len(np.unique(labels))
        pred_objects = len(np.unique(y_pred))
        print("Number of true objects:", true_objects-1)
        print("Number of predicted objects:", pred_objects-1)

        # Compute intersection between all objects
        intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

        # Compute areas (needed for finding the union between all objects)
        area_true = np.histogram(labels, bins = true_objects)[0]
        area_pred = np.histogram(y_pred, bins = pred_objects)[0]
        area_true = np.expand_dims(area_true, -1)
        area_pred = np.expand_dims(area_pred, 0)

        # Compute union
        union = area_true + area_pred - intersection

        # Exclude background from the analysis
        intersection = intersection[1:,1:]
        union = union[1:,1:]
        union[union == 0] = 1e-9

        # Compute the intersection over union
        iou = intersection / union
        dice = 2 * intersection / (union + intersection)
        return iou, dice
        
    def _3D_evaluation_2(self, target_study, pred_study):
        # TODO: double detected
        # target_nodules = target_study.nodule_instances
        pred_nodules = pred_study.nodule_instances
        pred_nodules_record = list(pred_nodules.keys())
        pred_nodules_record = {nodule_id: 'fp' for nodule_id in pred_nodules}
        tp, fp, fn = 0, 0, 0
        doubleCandidatesIgnored = 0

        iou, dsc = self.test(target_study.category_volume, pred_study.category_volume)
        max_dsc = np.max(dsc, axis=1)
        max_iou = np.max(iou, axis=1)
        if self.mean_iou is not None or self.mean_dsc is not None:
            self.mean_iou = np.concatenate((self.mean_iou, max_iou), axis=0)
            self.mean_dsc = np.concatenate((self.mean_dsc, max_dsc), axis=0)
        else:
            self.mean_iou = max_iou
            self.mean_dsc = max_dsc

        matchs = np.uint8(dsc>self.match_threshold)
        match_for_gt = np.sum(matchs, axis=1)
        tp = np.sum(np.where(match_for_gt>0, 1, 0))
        doubleCandidatesIgnored = np.sum(match_for_gt) - tp
        fp = matchs.shape[1] - tp - doubleCandidatesIgnored
        fn = np.sum(np.where(match_for_gt==0, 1, 0))

        for idx, num_match in enumerate(np.sum(matchs, axis=0)):
            if num_match > 1:
                pred_nodules_record[idx] = 'tp_double'
            elif num_match == 0:
                pred_nodules_record[idx] = 'fp'
            elif num_match == 1:
                pred_nodules_record[idx] = 'tp'

        # print(iou, dsc, np.max(dsc, axis=1))
        pred_study.nodule_evals.update(pred_nodules_record)

        target_study.set_score('NoduleTP', tp)
        target_study.set_score('NoduleFP', fp)
        target_study.set_score('NoduleFN', fn)

        self.VoxelTP.append(tp)
        self.VoxelFP.append(fp)
        self.VoxelFN.append(fn)
        self.DoubleDetections.append(doubleCandidatesIgnored)
        print(f'{target_study.study_id}: TP: {tp} FP: {fp} FN: {fn} Double Detected: {doubleCandidatesIgnored}')
        return tp, fp, fn, doubleCandidatesIgnored


    def _3D_evaluation(self, target_study, pred_study):
        # TODO: Best slice IoU
        target_nodules = target_study.nodule_instances
        pred_nodules = pred_study.nodule_instances
        pred_nodules_record = list(pred_nodules.keys())
        pp = {nodule_id: 'fp' for nodule_id in pred_nodules}
        tp, fp, fn = 0, 0, 0
        doubleCandidatesIgnored = 0

        iou, dsc = self.test(target_study.category_volume, pred_study.category_volume)
        print(iou, dsc, np.max(dsc, axis=1))
        
        for target_nodule_id in target_nodules:
            target_nodule = target_nodules[target_nodule_id]
            # NoduleIoU, NoduleDSC = [], []
            nodule_matches, nodule_match_scores = [], []
            found = False
            target_nodule.set_score('IoU', 0.0)
            target_nodule.set_score('DSC', 0.0)
            for pred_nodule_id in pred_nodules:
                pred_nodule = pred_nodules[pred_nodule_id]
                iou = self.BinaryIoU(target_nodule.nodule_volume, pred_nodule.nodule_volume)
                dsc = self.BinaryDSC(target_nodule.nodule_volume, pred_nodule.nodule_volume)

                def set_best_score(nodule, score_name, score):
                    score_in_nodule = nodule.get_score(score_name)
                    if score_in_nodule is not None:
                        if score > score_in_nodule:
                            nodule.set_score(score_name, score)
                    else:
                        nodule.set_score(score_name, score)

                set_best_score(pred_nodule, 'IoU', iou)
                set_best_score(pred_nodule, 'DSC', dsc)
                set_best_score(target_nodule, 'IoU', iou)
                set_best_score(target_nodule, 'DSC', dsc)

                if dsc > self.match_threshold:
                    found = True
                    nodule_matches.append(pred_nodule_id)
                    nodule_match_scores.append(dsc)
                    pp[pred_nodule_id] = 'tp'
                    # if pred_nodule_id in pred_nodules_record:
                    #     pred_nodules_record.remove(pred_nodule_id)
                    # else:
                    #     print(f'This is strange: CAD mark {pred_nodule_id} detected two nodules! Check for overlapping nodule annotations, SeriesUID: {target_study.study_id}, nodule Annot ID: {target_nodule_id}')
                    if 'tp' in pp[pred_nodule_id]:
                        print(f'This is strange: CAD mark {pred_nodule_id} detected two nodules! Check for overlapping nodule annotations, SeriesUID: {target_study.study_id}, nodule Annot ID: {target_nodule_id}')

            if len(nodule_matches) > 1: # double detection, single target be detected more than once
                doubleCandidatesIgnored += (len(nodule_matches) - 1)
                for n_id in nodule_matches:
                    pp[n_id] = 'tp_double'

            if found:
                tp += 1
                target_study.nodule_evals[target_nodule_id] = 'tp'
                pred_study.nodule_evals[pred_nodule_id] = 'tp'
            else:
                fn += 1
                target_study.nodule_evals[target_nodule_id] = 'fn'

        # fp = len(pred_nodules_record)
        # fp_num = list(pp.values()).count('fp')
        # for pred_nodule_id in pred_nodules_record:
        #     pred_study.nodule_evals[pred_nodule_id] = 'fp'
        fp = list(pp.values()).count('fp')
        pred_study.nodule_evals.update(pp)
        # TODO: change name pp to pred_nodules_record
        target_study.set_score('NoduleTP', tp)
        target_study.set_score('NoduleFP', fp)
        target_study.set_score('NoduleFN', fn)

        self.VoxelTP.append(tp)
        self.VoxelFP.append(fp)
        self.VoxelFN.append(fn)
        self.DoubleDetections.append(doubleCandidatesIgnored)
        print(f'{target_study.study_id}: TP: {tp} FP: {fp} FN: {fn} Double Detected: {doubleCandidatesIgnored}')
        return tp, fp, fn, doubleCandidatesIgnored

    def _2D_evaluation(self, target_study, pred_study):
        binary_target_vol = target_study.get_binary_volume()
        binary_pred_vol = pred_study.get_binary_volume()
        tp = np.sum(np.logical_and(binary_target_vol, binary_pred_vol))
        fp = np.sum(np.logical_and(np.logical_xor(binary_target_vol, binary_pred_vol), binary_pred_vol))
        fn = np.sum(np.logical_and(np.logical_xor(binary_target_vol, binary_pred_vol), binary_target_vol))

        iou = self.BinaryIoU(binary_target_vol, binary_pred_vol)
        dsc = self.BinaryDSC(binary_target_vol, binary_pred_vol)

        pred_study.set_score('Voxel IoU', iou)
        pred_study.set_score('Voxel DSC', dsc)

        self.PixelTP.append(tp)
        self.PixelFP.append(fp)
        self.PixelFN.append(fn)

    # @classmethod
    # def get_submission(cls, target_vol, pred_vol, vol_infos, connectivity=CONNECTIVITY, area_threshold=AREA_THRESHOLD):
    #     target_vol, target_metadata = cls.volume_preprocess(target_vol, connectivity, area_threshold)
    #     pred_vol, pred_metadata = cls.volume_preprocess(pred_vol, connectivity, area_threshold)
    #     pred_category = np.unique(pred_vol)[1:]
    #     total_nodule_infos = []
    #     for label in pred_category:
    #         pred_nodule = np.where(pred_vol==label, 1, 0)
    #         target_nodule = np.logical_or(target_vol>0, pred_nodule)*target_vol
    #         nodule_dsc = cls.BinaryDSC(target_nodule, pred_nodule)
    #         pred_center_irc = utils.get_nodule_center(pred_nodule)
    #         pred_center_xyz = utils.irc2xyz(pred_center_irc, vol_infos['origin'], vol_infos['spacing'], vol_infos['direction'])
    #         nodule_infos= {'Center_xyz': pred_center_xyz, 'Nodule_prob': nodule_dsc}
    #         total_nodule_infos.append(nodule_infos)
    #     return total_nodule_infos

    @staticmethod
    def BinaryIoU(target, pred):
        intersection = np.sum(np.logical_and(target, pred))
        union = np.sum(np.logical_or(target, pred))
        return intersection/union if union != 0 else 0
    
    @staticmethod
    def BinaryDSC(target, pred):
        intersection = np.sum(np.logical_and(target, pred))
        union = np.sum(np.logical_or(target, pred))
        return 2*intersection/(union+intersection) if (union+intersection) != 0 else 0

    def write_and_print(self, message):
        assert isinstance(message, str), 'Message is not string type.'
        self.eval_file.write(message + '\n')
        print(message)

    def evaluation(self, show_evaluation):
        self.VoxelTP = np.array(self.VoxelTP)
        self.VoxelFP = np.array(self.VoxelFP)
        self.VoxelFN = np.array(self.VoxelFN)
        self.P = self.VoxelTP + self.VoxelFN

        precision = self.VoxelTP / (self.VoxelTP + self.VoxelFP)
        recall = self.VoxelTP / (self.VoxelTP + self.VoxelFN)
        self.Precision = np.where(np.isnan(precision), 0, precision)
        self.Recall = np.where(np.isnan(recall), 0, recall)
        self.Volume_Precisiion = np.sum(self.VoxelTP) / (np.sum(self.VoxelTP) + np.sum(self.VoxelFP))
        self.Volume_Recall = np.sum(self.VoxelTP) / np.sum(self.P)
        self.Slice_Precision = np.sum(self.PixelTP) / (np.sum(self.PixelTP) + np.sum(self.PixelFP))
        self.Slice_Recall = np.sum(self.PixelTP) / (np.sum(self.PixelTP) + np.sum(self.PixelFN))
        self.Slice_F1 = 2*np.sum(self.PixelTP) / (2*np.sum(self.PixelTP) + np.sum(self.PixelFP) + np.sum(self.PixelFN))

        if show_evaluation:
            t = time.localtime()
            result = time.strftime("%m/%d/%Y, %H:%M:%S", t)
            self.write_and_print(f'Time: {result}')
            self.write_and_print(f'Model: {self.model_name}')
            self.write_and_print(f'Testing Dataset: {self.dataset_name}')
            self.write_and_print(f'Testing Case: {self.num_case}')
            self.write_and_print(f'Testing Nodule: {self.num_nodule}')
            self.write_and_print(f'Matching Threshold: {self.match_threshold}')
            self.write_and_print('')
            self.write_and_print(f'VoxelTP/Target: {np.sum(self.VoxelTP)}/{np.sum(self.P)}')
            self.write_and_print(f'VoxelTP/Prediction: {np.sum(self.VoxelTP)}/{np.sum(self.VoxelTP)+np.sum(self.VoxelFP)}')
            self.write_and_print(f'Voxel Precision: {self.Volume_Precisiion:.4f}')
            self.write_and_print(f'Voxel Recall: {self.Volume_Recall:.4f}')
            self.write_and_print(f'Double Detection: {sum(self.DoubleDetections)}')
            self.write_and_print('')
            self.write_and_print(f'PixelTP/Target: {np.sum(self.PixelTP)}/{np.sum(self.PixelTP)+np.sum(self.PixelFN)}')
            self.write_and_print(f'PixelTP/Prediction: {np.sum(self.PixelTP)}/{np.sum(self.PixelTP)+np.sum(self.PixelFP)}')
            self.write_and_print(f'Pixel Precision: {self.Slice_Precision:.4f}')
            self.write_and_print(f'Pixel Recall: {self.Slice_Recall:.4f}')
            self.write_and_print(f'Pixel F1: {self.Slice_F1:.4f}')
            self.write_and_print(f'mean IoU: {np.mean(self.mean_iou):.4f}')
            self.write_and_print(f'mean DSC: {np.mean(self.mean_dsc):.4f}')
            print('Mean DSC', self.mean_dsc)
            self.eval_file.close()
        
        return self.VoxelTP, self.VoxelFP, self.VoxelFN, self.Volume_Precisiion, self.Volume_Recall

