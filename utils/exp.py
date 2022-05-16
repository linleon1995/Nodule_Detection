import numpy as np
from data.data_utils import load_itk
import time
import os
import cv2
from dataset_conversion.build_coco import rle_decode, binary_mask_to_rle
import matplotlib.pyplot as plt
from dataset_conversion.rle import mask2rle



def str_vs_npy(times=1000):
    rle_path = rf'rle2.txt'
    with open(rle_path, 'r') as f:
        content = []
        for line in f:
            content.extend(line.split(' '))
    content_int = [int(x) for x in content]
    np.save('rle2.npy', np.array(content_int))

   
    start = time.time()
    for t in range(times):
        x = np.load('rle2.npy')
        shape = x[:8]
        semantic = x[2]
    end = time.time()
    rle_time = end-start
    print(f'RLE {rle_time} {rle_time:.4f} second')
    

def rle_3d_testing(times):
    for times_one in times:
        # preprocessing
        npy_path = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-preprocess\masks_test\3\1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.npy'
        mask = np.load(npy_path)
        rle_code = mask2rle(mask)
        with open('rle.txt', 'w+') as f:
            f.write(rle_code)

        # mask loading (rle)
        start = time.time()
        for i in range(times_one):
            with open('rle.txt', 'r') as f:
                rle_read_code = f.read()
            rle_mask = rle_decode(rle_read_code, mask.shape)
        end = time.time()
        rle_time = end-start
        print(f'RLE {times_one} {rle_time:.4f} second')

        # mask loading (cv2)
        npy_path = rf'C:\Users\test\Desktop\Leon\Datasets\LIDC-preprocess\masks_test\3\1.3.6.1.4.1.14519.5.2.1.6279.6001.100225287222365663678666836860.npy'
        start = time.time()
        for i in range(times_one):
            npy_mask = np.load(npy_path)
        end = time.time()
        npy_time = end-start
        print(f'Image {times_one} {npy_time:.4f} second')

        print(f'In {times_one} times test, RLE loading speed is faster {npy_time/rle_time:.2f} times than Image loading speed')
        binary_npy_mask = np.where(npy_mask>0, 1, 0)
        print(f'Same mask: {np.all(rle_mask==binary_npy_mask)}')
        print('\n')



def rle_testing(times):
    for times_one in times:
        cv2_path = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\image\Mask\TMH0005\TMH0005_0111.png'
        
        mask = cv2.imread(cv2_path)
        mask = mask[...,0]
        rle_code = mask2rle(mask)
        with open('rle.txt', 'w+') as f:
            f.write(rle_code)

        start = time.time()
        for i in range(times_one):
            with open('rle.txt', 'r') as f:
                rle_read_code = f.read()
            rle_mask = rle_decode(rle_read_code, mask.shape)
        end = time.time()
        rle_time = end-start
        print(f'RLE {times_one} {rle_time:.4f} second')

        # mask loading (cv2)
        cv2_path = rf'C:\Users\test\Desktop\Leon\Datasets\TMH_Nodule-preprocess\image\Mask\TMH0005\TMH0005_0111.png'
        start = time.time()
        for i in range(times_one):
            cv2_mask = cv2.imread(cv2_path)
        end = time.time()
        cv2_time = end-start
        print(f'Image {times_one} {cv2_time:.4f} second')

        print(f'In {times_one} times test, RLE loading speed is faster {cv2_time/rle_time:.2f} times than Image loading speed')
        print('\n')

    print(np.all(rle_mask==cv2_mask[...,0]))



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
    # f = rf'C:\Users\test\Desktop\Leon\Weekly\0428'
    # path_np = os.path.join(f, 'test.npy')
    # path_itk = os.path.join(f, '1.3.6.1.4.1.14519.5.2.1.6279.6001.102681962408431413578140925249.mhd')
    # test_loading_speed(path_np, path_itk)

    # times = [10**i for i in range(1, 4)]
    # rle_testing(times)
    # rle_3d_testing(times)

    str_vs_npy()


if __name__ == '__main__':
    main()

