'''
Modified Date: 2021/12/14
Author: Li-Wei Hsiao
mail: nfsmw308@gmail.com
'''
import collections
import copy
import cv2
import numpy as np

IrcTuple = collections.namedtuple('IrcTuple', ['index', 'row', 'col'])
XyzTuple = collections.namedtuple('XyzTuple', ['x', 'y', 'z'])

def irc2xyz(coord_irc, origin_xyz, vxSize_xyz, direction_a):
    cri_a = np.array(coord_irc)[::-1]
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coords_xyz = (direction_a @ (cri_a * vxSize_a)) + origin_a
    # coords_xyz = (direction_a @ (idx * vxSize_a)) + origin_a
    return XyzTuple(*coords_xyz)

def xyz2irc(coord_xyz, origin_xyz, vxSize_xyz, direction_a):
    origin_a = np.array(origin_xyz)
    vxSize_a = np.array(vxSize_xyz)
    coord_a = np.array(coord_xyz)
    cri_a = ((coord_a - origin_a) @ np.linalg.inv(direction_a)) / vxSize_a
    cri_a = np.round(cri_a)
    return IrcTuple(int(cri_a[2]), int(cri_a[1]), int(cri_a[0]))

class metrics:
    def __init__(self, n_class):
        self.n_class = n_class if n_class != 1 else 2
        self.class_intersection = np.zeros(self.n_class)
        self.class_union = np.zeros(self.n_class)
        self.class_gt_area = np.zeros(self.n_class)
        self.class_seg_area = np.zeros(self.n_class)
        # metrics
        self.class_acc = np.zeros(self.n_class)
        self.class_iou = np.zeros(self.n_class)
        self.class_f1 = np.zeros(self.n_class)
        self.pixel_TP = 0
        self.pixel_FP = 0
        self.pixel_FN = 0
        self.counter_TP = 0
        self.counter_FP = 0
        self.counter_FN = 0

    def Cal_area_2poly(self, img, original_bbox, prediction_bbox):
        im = np.zeros(img.shape[:], dtype = "uint8")
        im1 = np.zeros(img.shape[:], dtype = "uint8")
        original_grasp_mask = cv2.fillPoly(im, original_bbox.reshape((-1,original_bbox.shape[0],original_bbox.shape[2])), 255)
        prediction_grasp_mask = cv2.fillPoly(im1, prediction_bbox.reshape((-1,prediction_bbox.shape[0],prediction_bbox.shape[2])), 255)
        masked_and = cv2.bitwise_and(original_grasp_mask, prediction_grasp_mask, mask=im)
        masked_or = cv2.bitwise_or(original_grasp_mask, prediction_grasp_mask)
        or_area = np.sum(np.float32(np.greater(masked_or, 0)))
        and_area = np.sum(np.float32(np.greater(masked_and,0)))
        IOU = and_area / or_area
        return IOU

    def calculate(self, predicts, targets, iou_th = 0.5, area_th = 50):
        if targets.ndim == 4:
            gts = np.zeros((targets.shape[0], targets.shape[2], targets.shape[3]))
            for i in range(0, targets.shape[1]):
                gts[targets[:, i, :, :] == i] = i
            gts = gts.astype('int')
            targets = copy.deepcopy(gts)

        for seg, gt in zip(predicts, targets):
            for j in range(1, self.n_class):
                contours_gt, _ = cv2.findContours((gt==j).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours_seg, _ = cv2.findContours((seg==j).astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                contours_seg = [ seg for seg in contours_seg if cv2.contourArea(seg) > area_th]
                seg = cv2.fillPoly(np.zeros(seg.shape), contours_seg, j)
                
                self.class_intersection[j] += np.logical_and(gt==j,seg==j).sum()
                self.class_gt_area[j] += (gt==j).sum()
                self.class_seg_area[j] += (seg==j).sum()
                
                if j != 0:
                    for c_gt in contours_gt:
                        bad = True
                        for c_seg in contours_seg:
                            if self.Cal_area_2poly(gt, c_gt, c_seg) >= iou_th and cv2.contourArea(c_seg) > area_th:
                                self.counter_TP += 1
                                bad = False
                        if bad:
                            self.counter_FN += 1

                    for c_seg in contours_seg:
                        bad = True
                        # TODO: Why have this judgement? (redundant function)
                        # if cv2.contourArea(c_seg) > area_th:
                        #     continue
                        for c_gt in contours_gt:
                            if self.Cal_area_2poly(gt, c_gt, c_seg) >= iou_th:
                                bad = False
                        if bad:
                            self.counter_FP += 1

                self.pixel_TP += np.logical_and( gt==j, seg==j ).sum()
                self.pixel_FP += np.logical_and( np.logical_xor(gt==j, seg==j), (seg==j) ).sum()
                self.pixel_FN += np.logical_and( np.logical_xor(gt==j, seg==j), (gt==j) ).sum()

    def evaluation(self, show=False):
        for k in range(self.n_class):
            self.class_acc[k] = self.class_intersection[k]/self.class_gt_area[k] if self.class_gt_area[k] != 0 else 0
            self.class_iou[k] = self.class_intersection[k]/(self.class_gt_area[k]+self.class_seg_area[k]-self.class_intersection[k]) if (self.class_gt_area[k]+self.class_seg_area[k]-self.class_intersection[k]) != 0 else 0
            if (self.class_gt_area[k]+self.class_seg_area[k]) == 0 and 2*self.class_intersection[k] == 0:
                self.class_f1[k] = 1 if k > 0 else 0
            else:
                self.class_f1[k] = 2*self.class_intersection[k]/(self.class_gt_area[k]+self.class_seg_area[k])
       
        mIOU = self.class_iou[1:].mean()
        Total_dice = 2*self.class_intersection[1:].sum()/(self.class_gt_area[1:].sum()+self.class_seg_area[1:].sum()) if (self.class_gt_area[1:].sum()+self.class_seg_area[1:].sum()) != 0 else 0
        pixel_Precision = self.pixel_TP / (self.pixel_TP + self.pixel_FP) if (self.pixel_TP + self.pixel_FP) != 0 else 0
        pixel_Recall = self.pixel_TP / (self.pixel_TP + self.pixel_FN) if (self.pixel_TP + self.pixel_FN) != 0 else 0
        pixel_F1_score = 2*(pixel_Precision*pixel_Recall / (pixel_Recall + pixel_Precision)) if (pixel_Recall + pixel_Precision) != 0 else 0
        counter_Precision = self.counter_TP / (self.counter_TP + self.counter_FP) if (self.counter_TP + self.counter_FP) != 0 else 0
        counter_Recall = self.counter_TP / (self.counter_TP + self.counter_FN) if (self.counter_TP + self.counter_FN) != 0 else 0
        counter_F1_score = 2*(counter_Precision*counter_Recall / (counter_Recall + counter_Precision)) if (counter_Recall + counter_Precision) != 0 else 0
        if show:
            print('class accuracy =',self.class_acc)
            print('class IoU =',self.class_iou)
            print('class dice =',self.class_f1)
            print('Pixel Precision =',pixel_Precision)
            print('Pixel Recall =',pixel_Recall)
            print('Pixel F1 score =',pixel_F1_score)
            print('counter Precision =',counter_Precision)
            print('counter Recall =',counter_Recall)
            print('counter F1 score =',counter_F1_score)
            print('mIoU =', mIOU)
            print('total dice =', Total_dice)
        
        return self.class_acc, self.class_iou, self.class_f1, mIOU, pixel_Precision, pixel_Recall, Total_dice

