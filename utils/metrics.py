

from sqlalchemy import intersect
import numpy as np


def binary_dsc(target, pred):
    if np.sum(target)==0 and np.sum(pred)==0:
        return 1.0
    intersection = np.sum(target*pred)
    return (2*intersection) / (np.sum(target) + np.sum(pred))