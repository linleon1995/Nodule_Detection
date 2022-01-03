import numpy as np
import matplotlib.pyplot as plt
import cc3d
from volume_generator import luna16_volume_generator, lidc_volume_generator, asus_nodule_volume_generator

def nodule_based_acc(labels_in, connectivity):
    labels_out = cc3d.connected_components(labels_in, connectivity=connectivity)
    return labels_out

class volumetric_data_eval():
    def __init__(self, connectivity=26, area_threshold=20, match_threshold=0.5, max_nodule_num=1):
        self.connectivity = connectivity
        self.area_threshold = area_threshold
        self.match_threshold = match_threshold
        self.max_nodule_num = max_nodule_num
        self.TP, self.FP, self.FN = [], [] ,[]
    
    @staticmethod
    def remove_small_area(volume, area_threshold, max_nodule_num=None):
        # TODO: Could be faster (try using array ops, one-hot?)
        category = list(range(1, np.max(volume)+1))
        total_area_size = np.array([], dtype=np.int32)
        for label in range(1, np.max(volume)+1):
            area_size = np.sum(volume==label)
            if area_size < area_threshold:
                volume = np.where(volume==label, 0, volume)
                category.remove(label)
            else:
                total_area_size = np.append(total_area_size, area_size)

        # Descending sort the category based on nodule size
        category = np.take(category, np.argsort(total_area_size)[::-1])

        # Convert Label and keep big enough nodules (keeping nodule number = max_nodule_num)
        for idx, label in enumerate(category, 1):
            volume = np.where(volume==label, idx, volume)
            if max_nodule_num is not None:
                if idx == max_nodule_num:
                    break
        if max_nodule_num is not None:
            volume = np.where(volume>max_nodule_num, 0, volume)
        return volume

    def IoU(self, target, pred):
        # Only implement binary version
        intersection = np.sum(np.logical_and(target, pred))
        union = np.sum(np.logical_or(target, pred))
        return intersection/union if union != 0 else 0
    
    def calculate(self, mask_vol, pred_vol):
        mask_vol = cc3d.connected_components(mask_vol, connectivity=self.connectivity)
        pred_vol = cc3d.connected_components(pred_vol, connectivity=self.connectivity)
        mask_vol = self.remove_small_area(mask_vol, self.area_threshold)
        pred_vol = self.remove_small_area(pred_vol, self.area_threshold)
        assert np.shape(mask_vol) == np.shape(pred_vol)
        tp, fp, fn = 0, 0, 0

        mask_category = list(range(1, np.max(mask_vol)+1))
        pred_category = list(range(1, np.max(pred_vol)+1))
        for mask_label in mask_category:
            for pred_label in pred_category:
                if self.IoU(mask_vol==mask_label, pred_vol==pred_label) >= self.match_threshold:
                    tp += 1
                    mask_category.remove(mask_label)
                    pred_category.remove(pred_label)
                    break

        fp = len(pred_category)
        fn = len(mask_category)
        self.TP.append(tp)
        self.FP.append(fp)
        self.FN.append(fn)

    def evaluation(self, show_evaluation):
        self.TP = np.array(self.TP)
        self.FP = np.array(self.FP)
        self.FN = np.array(self.FN)
        self.P = self.TP + self.FN
        precision = self.TP / (self.TP + self.FP)
        recall = self.TP / (self.TP + self.FN)
        self.Precision = np.where(np.isnan(precision), 0, precision)
        self.Recall = np.where(np.isnan(recall), 0, recall)
        self.System_Precisiion = np.sum(self.TP) / (np.sum(self.TP) + np.sum(self.FP))
        self.System_Recall = np.sum(self.TP) / np.sum(self.P)

        if show_evaluation:
            print(f'TP / Target: {np.sum(self.TP)} / {np.sum(self.P)}')
            print(f'FN / Target: {np.sum(self.FN)} / {np.sum(self.P)}')
            print(f'TP / Prediction: {np.sum(self.TP)} / {np.sum(self.TP)+np.sum(self.FP)}')
            print(f'FP / Prediction: {np.sum(self.FP)} / {np.sum(self.TP)+np.sum(self.FP)}')
            print('')
            print(f'System Precision: {self.System_Precisiion:.4f}')
            print(f'System Recall: {self.System_Recall:.4f}')
            print('')
            print(f'max Precision: {np.max(self.Precision):.4f}')
            print(f'min Precision: {np.min(self.Precision):.4f}')
            print(f'mean Precision: {np.mean(self.Precision):.4f}')
            print(f'max Recall: {np.max(self.Recall):.4f}')
            print(f'min Recall: {np.min(self.Recall):.4f}')
            print(f'mean Recall: {np.mean(self.Recall):.4f}')
        return self.TP, self.FP, self.FN, self.System_Precisiion, self.System_Recall


def main():
    data_path = rf'C:\Users\test\Desktop\Leon\Datasets\LUNA16\data'
    volume_generator = luna16_volume_generator(data_path)
    v = 5

    for _, labels_in, _  in volume_generator:
        labels_out = nodule_based_acc(labels_in, 26)

        print('IN')
        for i in range(1, v+1):
            print(i, np.sum(labels_in==i))

        print('OUT')
        for i in range(1, v+1):
            print(i, np.sum(labels_out==i))
            if i > 1 and np.sum(labels_out==i) > 0:
                indices = np.where(labels_out==2)
                plt.imshow(labels_out[indices[0][0]])
                plt.show()
                plt.imshow(labels_out[indices[0][0]-1])
                plt.show()
                plt.imshow(labels_out[indices[0][0]+1])
                plt.show()
                print(indices)

if __name__ == '__main__':
    main()