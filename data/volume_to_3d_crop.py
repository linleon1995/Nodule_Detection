

# TODO: consider overlapping
import numpy as np


class CropVolume():
    def __init__(self, crop_size, crop_shift):
        self.crop_size = crop_size
        self.crop_shift = crop_shift

    def __call__(self, volume):
        crop_data = []
        crop_slices = self.get_crop_slice(volume.shape)
        
        for slice_range in crop_slices:
            # crop = volume.copy()
            # for dim , crop_slice_dim in enumerate(crop_slice):
            #     crop = np.take(crop, crop_slice_dim, dim)
            crop_slice = [slice(*r) for r in slice_range]
            crop = volume[crop_slice]
            crop_data.append({'slice': slice_range, 'data':crop})
        return crop_data

    def get_crop_slice(self, volume_shape):
        slices = []
        for dim_idx, length in enumerate(volume_shape):
            slices.append(self.simple_slice(length, self.crop_shift[dim_idx], self.crop_size[dim_idx]))

        # TODO: for arbitrary dimensions
        slice_comb = []
        for h in slices[0]:
            for w in slices[1]:
                for c in slices[2]: 
                    slice_comb.append([h, w, c])

        # def comb(total_slices, slices, dim):
        #     if dim == len(volume_shape)+1:
        #         return total_slices
        #     else:
        #         dim += 1
        #         for slice in total_slices:
        #             comb([slice], slices, dim)
        #         # comb(total_slices, slices, dim)
        # total_slices = comb(slices[0], slices[1:], 0)
        return slice_comb

    @staticmethod
    def simple_slice(length, shift, crop_length):
        slices = []
        for start in range(shift, length-shift-crop_length, crop_length):
            # slices.append(slice(start, start+crop_length))
            slices.append((start, start+crop_length))
        return slices


if __name__ == '__main__':
    # import numpy as np

    crop_ops = CropVolume((32,64,64), (0, 100, 100))
    np_data = np.zeros((100, 512, 512))
    a = np_data[1:30, 2:45]
    b = slice(*(1,51))
    # a_idx = np.arange(1, 10)
    # b_idx = np.arange(11, 16)
    # c_idx = np.meshgrid(a_idx, b_idx)
    # a = np_data[a_idx]
    # b = np_data[:,b_idx,b_idx]
    # z_indices = np.random.randint(0,30,(30,30))

    # def linidx_take(val_arr,z_indices):
    #     # Get number of columns and rows in values array
    #     _,nC,nR = val_arr.shape

    #     # Get linear indices and thus extract elements with np.take
    #     idx = nC*nR*z_indices + nR*np.arange(nR)[:,None] + np.arange(nC)
    #     return np.take(val_arr,idx) # Or val_arr.ravel()[idx]
        
    # c = np.take(np_data, z_indices)
    crop_ops(np_data)
    print(3)
    