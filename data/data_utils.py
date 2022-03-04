import numpy as np
import cc3d


def cv2_imshow(img, save_path=None):
    # pass
    cv2.imshow('My Image', img)
    cv2.imwrite(save_path if save_path else 'sample.png', img)
    # cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
def split_individual_mask(mask):
    individual_mask = cc3d.connected_components(mask, connectivity=8)
    if np.max(individual_mask) > 1:
        mask_list = []
        for mask_label in range(1, np.max(individual_mask)):
            mask_list.append(np.uint8(individual_mask==mask_label))
        return mask_list
    else:
        return [mask]


def merge_near_masks(sub_masks, distance_threshold=128):
    candidate_pool = []
    for mask in sub_masks:
        if not candidate_pool:
            candidate_pool.append(mask)
        else:
            for candidate_idx, mask_candidate in enumerate(candidate_pool):
                ys, xs = np.where(mask)
                ys2, xs2 = np.where(mask_candidate)
                mask_min_point = np.array([np.min(ys), np.min(xs)])
                candidate_min_point = np.array([np.min(ys2), np.min(xs2)])
                mask_distance =  np.linalg.norm(mask_min_point-candidate_min_point)
                # print('distance', mask_distance)
                if mask_distance < distance_threshold:
                    candidate_pool[candidate_idx] = mask_candidate + mask
                    append_flag = False
                    break
                else:
                    append_flag = True
            
            if append_flag:
                candidate_pool.append(mask)
    return candidate_pool