import numpy as np
import matplotlib.pyplot as plt
import cc3d

class volumetric_data_eval():
    def __init__(self, connectivity=26, area_threshold=20, match_threshold=0.5, max_nodule_num=1):
        self.connectivity = connectivity
        self.area_threshold = area_threshold
        self.match_threshold = match_threshold
        self.max_nodule_num = max_nodule_num
        self.PixelTP, self.PixelFP, self.PixelFN = [], [] ,[]
        self.VoxelTP, self.VoxelFP, self.VoxelFN = [], [] ,[]

    def calculate(self, mask_vol, pred_vol):
        mask_vol = self.volume_preprocess(mask_vol, self.connectivity, self.area_threshold)
        pred_vol = self.volume_preprocess(pred_vol, self.connectivity, self.area_threshold)
        assert np.shape(mask_vol) == np.shape(pred_vol)
        nodule_infos = self._3D_evaluation(mask_vol, pred_vol)
        self._2D_evaluation(mask_vol, pred_vol)
        return nodule_infos
    
    @classmethod
    def volume_preprocess(cls, volume, connectivity, area_threshold):
        volume = cc3d.connected_components(volume, connectivity=connectivity)
        volume = cls.remove_small_area(volume, area_threshold)
        return volume

    def _3D_evaluation(self, mask_vol, pred_vol):
        tp, fp, fn = 0, 0, 0
        mask_category = list(range(1, np.max(mask_vol)+1))
        pred_category = list(range(1, np.max(pred_vol)+1))
        total_nodule_infos = []

        for nodule_idx, mask_label in enumerate(range(1, np.max(mask_vol)+1)):
            gt_nodule = np.int32(mask_vol==mask_label)
            gt_nodule_size = np.sum(gt_nodule)
            BestNoduleIoU, BestNoduleDSC = 0, 0
            BestSliceIoU, BestSliceIndex = 0, 'Null'
            nodule_infos = {}
            # TODO: calculate_nodule_slice will cause confusion (list or dict?)
            mask_nonzero_slice = self.calculate_nodule_slice(gt_nodule)[1]
            slice_num = mask_nonzero_slice[1] - mask_nonzero_slice[0] + 1
            nodule_infos = {'Nodule ID': np.int32(nodule_idx), 'Nodule IoU': 0, 'Nodule DSC': 0, 
                            'Slice Number': np.int32(slice_num), 
                            'Size': np.int32(gt_nodule_size), 'Relative Size': gt_nodule_size/mask_vol.size,
                            'Best Slice IoU': BestSliceIoU, 'Best Slice Index': BestSliceIndex}
            for pred_label in range(1, np.max(pred_vol)+1):
                pred_nodule = np.int32(pred_vol==pred_label)
                NoduleIoU = self.IoU(gt_nodule, pred_nodule)
                NoduleDSC = self.DSC(gt_nodule, pred_nodule)

                if NoduleIoU > 0:
                    if NoduleIoU > BestNoduleIoU:
                        BestNoduleIoU = NoduleIoU
                        BestNoduleDSC = NoduleDSC
                        BestSliceIoU, BestSliceIndex = 0, 'Null'
                        for slice_idx in range(mask_nonzero_slice[0], mask_nonzero_slice[1]+1):
                            SliceIOU = self.IoU(mask_vol[slice_idx]==mask_label, pred_vol[slice_idx]==pred_label)
                            if SliceIOU > BestSliceIoU:
                                BestSliceIoU = SliceIOU
                                BestSliceIndex = slice_idx

                    if NoduleIoU >= self.match_threshold:
                        tp += 1
                        mask_category.remove(mask_label)
                        pred_category.remove(pred_label)
                        break

            nodule_infos['Nodule IoU'] = BestNoduleIoU
            nodule_infos['Nodule DSC'] = BestNoduleDSC
            nodule_infos['Best Slice IoU'] = BestSliceIoU
            nodule_infos['Best Slice Index'] = BestSliceIndex
            total_nodule_infos.append(nodule_infos)

        fp = len(pred_category)
        fn = len(mask_category)
        self.VoxelTP.append(tp)
        self.VoxelFP.append(fp)
        self.VoxelFN.append(fn)
        return total_nodule_infos
    
    def _2D_evaluation(self, mask_vol, pred_vol):
        binary_mask_vol = np.where(mask_vol>0, 1, 0)
        binary_pred_vol = np.where(pred_vol>0, 1, 0)
        tp = np.sum(np.logical_and(binary_mask_vol, binary_pred_vol))
        fp = np.sum(np.logical_and(np.logical_xor(binary_mask_vol, binary_pred_vol), binary_pred_vol))
        fn = np.sum(np.logical_and(np.logical_xor(binary_mask_vol, binary_pred_vol), binary_mask_vol))
        self.PixelTP.append(tp)
        self.PixelFP.append(fp)
        self.PixelFN.append(fn)

    @classmethod
    def remove_small_area(cls, volume, area_threshold, max_nodule_num=None):
        """The nodule order in not promissed"""
        total_area_size, volume = cls.calculate_area_size_and_remove_small_one_in_volume(volume, area_threshold)

        # Descending sort the category based on nodule size
        # category = list(range(1, np.max(volume)+1))
        category = np.unique(volume)
        category = np.delete(category, np.where(category==0))
        # category = np.take(category, np.argsort(total_area_size)[::-1])

        # Convert Label and keep big enough nodules (keeping nodule number = max_nodule_num)
        new_volume = np.zeros_like(volume)
        for idx, label in enumerate(category, 1):
            # volume = np.where(volume==label, idx, volume)
            new_volume[volume==label] = idx
            # print(np.unique(new_volume))
            if max_nodule_num is not None:
                if idx == max_nodule_num:
                    break
        volume = new_volume

        if max_nodule_num is not None:
            volume = np.where(volume>max_nodule_num, 0, volume)
        # print(np.unique(volume))
        # print(np.max(volume))
        return volume

    @classmethod
    def calculate_area_size_and_remove_small_one_in_volume(cls, volume, area_threshold):
        """Calculate every nodule size in single volume and return nodule sizes bigger than area_threshold"""
        # TODO: separate duty, caluculate size, change volume value
        # _, counts = np.unique(volume, return_counts=True)
        # total_area_size = np.fromiter(counts.values(), dtype=int)[1:]
        # total_area_size = np.delete(total_area_size, np.where(total_area_size<area_threshold)[0])
        # print(np.unique(volume))
        category = list(range(1, np.max(volume)+1))
        total_area_size = np.array([], dtype=np.int32)
        for label in range(1, np.max(volume)+1):
            area_size = np.sum(volume==label)
            if area_size < area_threshold:
                volume = np.where(volume==label, 0, volume)
                category.remove(label)
            else:
                total_area_size = np.append(total_area_size, area_size)
        # print(np.unique(volume))
        return total_area_size, volume

    # @classmethod
    # def convert_volume_value(cls, volume, ):

    @staticmethod
    def calculate_nodule_slice(volume):
        # TODO: make sure input is bool or int array
        if np.sum(volume) == np.sum(np.zeros_like(volume)):
            return None

        category = np.unique(volume)
        nodule_size = {}
        for label in category:
            zs, ys, xs = np.where(volume==label)
            nodule_size[label] = (np.min(zs), np.max(zs))
        return nodule_size

    @staticmethod
    def IoU(target, pred):
        # TODO: Add constraint for dtype or build multi-class version
        # Only input binary array
        intersection = np.sum(np.logical_and(target, pred))
        union = np.sum(np.logical_or(target, pred))
        return intersection/union if union != 0 else 0
    
    @staticmethod
    def DSC(target, pred):
        intersection = np.sum(np.logical_and(target, pred))
        union = np.sum(np.logical_or(target, pred))
        return 2*intersection/(union+intersection) if (union+intersection) != 0 else 0

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
            print(f'Area Threshold: {self.area_threshold}')
            print(f'VoxelTP / Target: {np.sum(self.VoxelTP)} / {np.sum(self.P)}')
            print(f'VoxelFN / Target: {np.sum(self.VoxelFN)} / {np.sum(self.P)}')
            print(f'VoxelTP / Prediction: {np.sum(self.VoxelTP)} / {np.sum(self.VoxelTP)+np.sum(self.VoxelFP)}')
            print(f'VoxelFP / Prediction: {np.sum(self.VoxelFP)} / {np.sum(self.VoxelTP)+np.sum(self.VoxelFP)}')
            print(f'System Precision: {self.Volume_Precisiion:.4f}')
            print(f'System Recall: {self.Volume_Recall:.4f}')
            print('')
            print(f'PixelTP / Target: {np.sum(self.PixelTP)} / {np.sum(self.PixelTP)+np.sum(self.PixelFN)}')
            print(f'PixelFN / Target: {np.sum(self.PixelFN)} / {np.sum(self.PixelTP)+np.sum(self.PixelFN)}')
            print(f'PixelTP / Prediction: {np.sum(self.PixelTP)} / {np.sum(self.PixelTP)+np.sum(self.PixelFP)}')
            print(f'PixelFP / Prediction: {np.sum(self.PixelFP)} / {np.sum(self.PixelTP)+np.sum(self.PixelFP)}')
            print(f'Pixel Precision: {self.Slice_Precision:.4f}')
            print(f'Pixel Recall: {self.Slice_Recall:.4f}')
            print(f'Pixel F1: {self.Slice_F1:.4f}')
            print('')
            print(f'mean Precision: {np.mean(self.Precision):.4f}')
            print(f'mean Recall: {np.mean(self.Recall):.4f}')
        return self.VoxelTP, self.VoxelFP, self.VoxelFN, self.Volume_Precisiion, self.Volume_Recall