import numpy as np
from data.data_utils import load_itk
import time
import os

def test_loading_speed(path_np, path_itk, times=100):
    np_start = time.time()
    for i in range(times):
        x = np.load(path_np)
    np_end = time.time()
    np_time = np_end-np_start
    print('Numpy',np_time)

    itk_start = time.time()
    for i in range(times):
        x = load_itk(path_itk)
    itk_end = time.time()
    itk_time = itk_end-itk_start
    print('ITK', itk_time)
    print(f'{abs(itk_time-np_time)/max(itk_time, np_time)*100:.2f} %')

def main():
    f = rf'C:\Users\test\Desktop\Leon\Weekly\0428'
    path_np = os.path.join(f, 'test.npy')
    path_itk = os.path.join(f, '1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249.mhd')
    test_loading_speed(path_np, path_itk)


if __name__ == '__main__':
    main()

