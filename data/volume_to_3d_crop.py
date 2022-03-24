

# TODO: consider overlapping
from lib2to3.pgen2.literals import simple_escapes


class CropVolume():
    def __init__(self, crop_size, crop_shift):
        self.crop_size = crop_size
        self.crop_shift = crop_shift

    def __call__(self, volume):
        crop_data = []
        crop_slices = self.get_crop_slice(volume.shape)
        for crop_slice in crop_slices:
            crop = volume[crop_slice]
            crop_data.append(crop)
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
            slices.append(slice(start, start+crop_length))
        return slices


if __name__ == '__main__':
    import numpy as np

    crop_ops = CropVolume((32,64,64), (0, 100, 100))
    np_data = np.zeros((100, 512, 512))
    crop_ops(np_data)
    