import cv2
import numpy as np
from data import transform_utils
scale_to_limit_size = transform_utils.scale_to_limit_size
random_flip = transform_utils.random_flip
random_scale = transform_utils.random_scale
random_rotate = transform_utils.random_rotate
gaussian_blur = transform_utils.gaussian_blur
random_crop = transform_utils.random_crop
show_data = transform_utils.show_data
random_gamma = transform_utils.random_gamma
random_gaussian = transform_utils.random_gaussian


# TODO: default params
# TODO: test image only, label only, image_label pair
# TODO: params check
# TODO: check deeplab code for generalization
# TODO: rectangle cropping
# TODO: rectangle resize?
class ImageDataTransformer():
    def __init__(self, preprocess_config):
        self.RandFlip = preprocess_config.get('RandFlip', None)
        self.RandCrop = preprocess_config.get('RandCrop', None)
        self.RandScale = preprocess_config.get('RandScale', None)
        self.PadToSquare = preprocess_config.get('PadToSquare', None)
        self.ScaleToSize = preprocess_config.get('ScaleToSize', None)
        self.ScaleLimitSize = preprocess_config.get('ScaleLimitSize', None)
        self.RandRotate = preprocess_config.get('RandRotate', None)
        self.GaussianBlur = preprocess_config.get('GaussianBlur', None)
        self.RandGamma = preprocess_config.get('RandGamma', None)
        self.RandGaussian = preprocess_config.get('RandGaussian', None)

        self.padding_height = preprocess_config.get('padding_height', None)
        self.padding_width = preprocess_config.get('padding_width', None)
        self.padding_value = preprocess_config.get('padding_value', None)
        self.flip_prob = preprocess_config.get('flip_prob', None)
        self.flip_mode = preprocess_config.get('flip_mode', None)
        self.min_scale_factor = preprocess_config.get('min_scale_factor', None)
        self.max_scale_factor = preprocess_config.get('max_scale_factor', None)
        self.step_size = preprocess_config.get('step_size', None)
        assert (preprocess_config.get('resize_method', None) == 'Bilinear' or preprocess_config.get('resize_method', None) == 'Cubic')
        self.resize_method = cv2.INTER_LINEAR if preprocess_config.get('resize_method', None) == 'Bilinear' else cv2.INTER_CUBIC
        self.crop_size = preprocess_config.get('crop_size', None)
        # self.scale_size = preprocess_config.get('scale_size', None)
        self.min_angle = preprocess_config.get('min_angle', None)
        self.max_angle = preprocess_config.get('max_angle', None)
        self.show_preprocess = preprocess_config.get('show_preprocess', None)
        self.min_gamma_factor = preprocess_config.get('min_gamma_factor', None)
        self.max_gamma_factor = preprocess_config.get('max_gamma_factor', None)
        self.gamma_factor_step_size = preprocess_config.get('gamma_factor_step_size', None)
        self.min_std_factor = preprocess_config.get('min_std_factor', None)
        self.max_std_factor = preprocess_config.get('max_std_factor', None)
        self.std_step_size = preprocess_config.get('std_step_size', None)

    def __call__(self, image, label=None):
        image, label = self.data_preprocess(image), self.data_preprocess(label)
        
        self.original_image = image
        self.original_label = label
        H = image.shape[0]
        W = image.shape[1]
        Hs, Ws = H, W
        image = np.squeeze(image)
        if label is not None:
            label = np.squeeze(label)

        if self.show_preprocess:
            print('method: original')
            show_data(image, label)

        if self.RandScale:
            image, label = random_scale(image, label, 
                self.min_scale_factor, self.max_scale_factor, self.step_size, self.resize_method)

        if self.RandRotate:
            image, label = random_rotate(image, label, 
                min_angle=self.min_angle, max_angle=self.max_angle)

        if self.RandGamma:
            image, label = random_gamma(image, label, 
                self.min_gamma_factor, self.max_gamma_factor, self.gamma_factor_step_size)

        if self.RandGaussian:
            image, label = random_gaussian(image, label,
                self.min_std_factor, self.max_std_factor, self.std_step_size)

        # if self.GaussianBlur:
        #     image, label = gaussian_blur(image, label)

        # if self.RandCrop:
        #     Ws = np.random.randint(0, Ws - self.crop_size[1] + 1, 1)[0]
        #     Hs = np.random.randint(0, Hs - self.crop_size[0] + 1, 1)[0]
        #     image = image[Hs:Hs + self.crop_size[0], Ws:Ws + self.crop_size[0]]
        #     if label is not None:
        #         label = label[Hs:Hs + self.crop_size[1], Ws:Ws + self.crop_size[1]]
        #     if SHOW_PREPROCESSING: 
        #         show_data_information(image, label, 'random crop')

        if len(image.shape)==2: image = image[...,np.newaxis]
        if label is not None:
            if len(label.shape)==2: label = label[...,np.newaxis]
        return (image, label)


    def data_preprocess(self, data):
        """Modift data type and data shape"""
        return data