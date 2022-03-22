import numpy as np
import cv2
import matplotlib.pyplot as plt
import functools
SHOW_PREPROCESSING = False


def show_data(*args, **kargs):
    # TODO: Dynamic for single input function
    pair = []
    pair.append(kargs['image']) if 'image' in kargs else pair.append(args[0])
    pair.append(kargs['label']) if 'label' in kargs else pair.append(args[1])
    image, label = pair
    print(f'    Image (value) max={np.max(image)} min={np.min(image)}  (shape) {image.shape}')
    if label is not None:
        print(f'    Label (value) max={np.max(image)} min={np.min(image)}  (shape) {image.shape}')
    _, (ax1, ax2) = plt.subplots(1,2)
    ax1.imshow(np.uint8(image), 'gray')
    if label is not None:
        ax2.imshow(np.uint8(label), 'gray')
    plt.show()


def show_data_information(show_preprocessing, op_name):
    def decorator(f):
        def called(*args, **kargs):
            if show_preprocessing:
                print(f'method: {op_name}')
                show_data(*args, **kargs)
            return f(*args, **kargs)
        return called
    return decorator


# TODO: Understand the code
# TODO: decorator
@show_data_information(SHOW_PREPROCESSING, op_name='resize to range')
def resize_to_range(image,
                    label=None,
                    min_size=None,
                    max_size=None,
                    factor=None,
                    label_layout_is_chw=False,
                    method=cv2.INTER_LINEAR):
    """Resizes image or label so their sides are within the provided range.

    The output size can be described by two cases:
    1. If the image can be rescaled so its minimum size is equal to min_size
        without the other side exceeding max_size, then do so.
    2. Otherwise, resize so the largest side is equal to max_size.

    An integer in `range(factor)` is added to the computed sides so that the
    final dimensions are multiples of `factor` plus one.

    Args:
        image: A 3D tensor of shape [height, width, channels].
        label: (optional) A 3D tensor of shape [height, width, channels] (default)
        or [channels, height, width] when label_layout_is_chw = True.
        min_size: (scalar) desired size of the smaller image side.
        max_size: (scalar) maximum allowed size of the larger image side. Note
        that the output dimension is no larger than max_size and may be slightly
        smaller than min_size when factor is not None.
        factor: Make output size multiple of factor plus one.
        align_corners: If True, exactly align all 4 corners of input and output.
        label_layout_is_chw: If true, the label has shape [channel, height, width].
        We support this case because for some instance segmentation dataset, the
        instance segmentation is saved as [num_instances, height, width].
        scope: Optional name scope.
        method: Image resize method. Defaults to tf.image.ResizeMethod.BILINEAR.

    Returns:
        A 3-D tensor of shape [new_height, new_width, channels], where the image
        has been resized (with the specified method) so that
        min(new_height, new_width) == ceil(min_size) or
        max(new_height, new_width) == ceil(max_size).

    Raises:
        ValueError: If the image is not a 3D tensor.
    """
    new_tensor_list = []
    min_size = float(min_size)
    if max_size is not None:
        max_size = float(max_size)
        # Modify the max_size to be a multiple of factor plus 1 and make sure the
        # max dimension after resizing is no larger than max_size.
        if factor is not None:
            max_size = (max_size + (factor - (max_size - 1) % factor) % factor
                        - factor)

    [orig_height, orig_width, _] = image.shape
    orig_height = float(orig_height)
    orig_width = float(orig_width)
    orig_min_size = min(orig_height, orig_width)

    # Calculate the larger of the possible sizes
    large_scale_factor = min_size / orig_min_size
    large_height = int(orig_height * large_scale_factor) + 1
    large_width = int(orig_width * large_scale_factor) + 1
    large_size = np.stack([large_width, large_height])
    
    new_size = large_size
    if max_size is not None:
        # Calculate the smaller of the possible sizes, use that if the larger
        # is too big.
        orig_max_size = max(orig_height, orig_width)
        small_scale_factor = max_size / orig_max_size
        small_height = int(orig_height * small_scale_factor) + 1
        small_width = int(orig_width * small_scale_factor) + 1
        small_size = np.stack([small_width, small_height])
        new_size = small_size if float(np.max(large_size)) > max_size else large_size
        # new_size = np.cond(
        #     float(np.max(large_size)) > max_size,
        #     lambda: small_size,
        #     lambda: large_size)
    # Ensure that both output sides are multiples of factor plus one.
    if factor is not None:
        new_size += (factor - (new_size - 1) % factor) % factor
    image = cv2.resize(image, (new_size[0], new_size[1]), interpolation=method)
    if len(image.shape)==2: image = image[...,np.newaxis]
    new_tensor_list.append(image)
    # new_tensor_list.append(tf.image.resize_images(
    #     image, new_size, method=method, align_corners=align_corners))
    if label is not None:
        if label_layout_is_chw:
            # Input label has shape [channel, height, width].
            resized_label = np.expand_dims(label, 3)
            resized_label = cv2.resize(resized_label, (new_size[0], new_size[1]), interpolation=cv2.INTER_NEAREST)
            # resized_label = tf.image.resize_nearest_neighbor(
            #     resized_label, new_size, align_corners=align_corners)
            resized_label = np.squeeze(resized_label, 3)
        else:
            # Input label has shape [height, width, channel].
            resized_label = cv2.resize(label, (new_size[0], new_size[1]), interpolation=cv2.INTER_NEAREST)
            # resized_label = tf.image.resize_images(
            #     label, new_size, method=cv2.INTER_NEARST,
            #     align_corners=align_corners)
        if len(resized_label.shape)==2: resized_label = resized_label[...,np.newaxis]
        new_tensor_list.append(resized_label)
    else:
        new_tensor_list.append(None)
    return new_tensor_list

# TODO: Add decorator
def pad_to_bounding_box(image, offset_height, offset_width, target_height, target_width, pad_value):
    """Pads the given image with the given pad_value.

    Works like tf.image.pad_to_bounding_box, except it can pad the image
    with any given arbitrary pad value and also handle images whose sizes are not
    known during graph construction.

    Args:
        image: 3-D tensor with shape [height, width, channels]
        offset_height: Number of rows of zeros to add on top.
        offset_width: Number of columns of zeros to add on the left.
        target_height: Height of output image.
        target_width: Width of output image.
        pad_value: Value to pad the image tensor with.

    Returns:
        3-D tensor of shape [target_height, target_width, channels].

    Raises:
        ValueError: If the shape of image is incompatible with the offset_* or
        target_* arguments.
    """
    image_rank = len(image.shape)
    image_shape = np.shape(image)
    height, width = image_shape[0], image_shape[1]
    assert image_rank == 3, 'Wrong image tensor rank, expected 3'
    assert target_width >= width, 'target_width must be >= width'
    assert target_height >= height, 'target_height must be >= height'
    after_padding_width = target_width - offset_width - width
    after_padding_height = target_height - offset_height - height
    assert (after_padding_width >= 0 and after_padding_height >= 0), \
        'target size not possible with the given target offsets'

    paddings = (after_padding_height, 0, after_padding_width, 0)
    padded = cv2.copyMakeBorder(image, *paddings, cv2.BORDER_CONSTANT, value=pad_value)
    return padded


def standardize(m, mean=None, std=None, eps=1e-10, channelwise=False, *kwargs):
    if mean is None:
        if channelwise:
            # normalize per-channel
            axes = list(range(m.ndim))
            # average across channels
            axes = tuple(axes[1:])
            mean = np.mean(m, axis=axes, keepdims=True)
            std = np.std(m, axis=axes, keepdims=True)
        else:
            mean = np.mean(m)
            std = np.std(m)
    num = (m - mean) / np.clip(std, a_min=eps, a_max=None)
    print(np.mean(num), np.std(num))
    return num


@show_data_information(SHOW_PREPROCESSING, op_name='gamma')
def random_gamma(image, label=None, min_gamma_factor=0.9, max_gamma_factor=1.1, 
                 gamma_factor_step_size=0.01):
    def gamma_transform(image, gamma):
        if image.dtype == np.uint8: image = np.float32(image/255)
        image = np.power(image, gamma)
        image = np.uint8(image*255)
        return image
    gamma = get_random_scale(min_gamma_factor, max_gamma_factor, gamma_factor_step_size)
    image = gamma_transform(image, gamma)
    if label is not None:
        label = gamma_transform(label, gamma)
    return image, label
    

def output_strides_align(input_image, output_strides, gt_image=None):
    H = input_image.shape[0]
    W = input_image.shape[1]
    top, left = (output_strides-H%output_strides)//2, (output_strides-W%output_strides)//2
    bottom, right = (output_strides-H%output_strides)-top, (output_strides-W%output_strides)-left
    input_image = cv2.copyMakeBorder(input_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0.0)
    if gt_image is not None:
        gt_image = cv2.copyMakeBorder(gt_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0.0)
    return input_image, gt_image


def get_random_scale(min_scale_factor, max_scale_factor, step_size):
    """Gets a random scale value.
    Args:
        min_scale_factor: Minimum scale value.
        max_scale_factor: Maximum scale value.
        step_size: The step size from minimum to maximum value.
    Returns:
        A random scale value selected between minimum and maximum value.
    Raises:
        ValueError: min_scale_factor has unexpected value.
    """
    if min_scale_factor < 0 or min_scale_factor > max_scale_factor:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_scale_factor == max_scale_factor:
        return float(min_scale_factor)

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return np.random.uniform(min_scale_factor, max_scale_factor)

    # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_scale_factor - min_scale_factor) / step_size + 1)
    scale_factors = list(np.linspace(min_scale_factor, max_scale_factor, num_steps))
    return scale_factors[np.random.randint(0, len(scale_factors))]


def get_random_uniform_value(min_value, max_value, step_size):
    """Refer to DeepLab get_random_scale"""
    if min_value > max_value:
        raise ValueError('Unexpected value of min_scale_factor.')

    if min_value == max_value:
        return float(min_value)

    # When step_size = 0, we sample the value uniformly from [min, max).
    if step_size == 0:
        return np.random.uniform(min_value, max_value)

    # When step_size != 0, we randomly select one discrete value from [min, max].
    num_steps = int((max_value - min_value) / step_size + 1)
    scale_factors = list(np.linspace(min_value, max_value, num_steps))
    return scale_factors[np.random.randint(0, len(scale_factors))]


def scale_to_limit_size(image, label, crop_size, resize_method=cv2.INTER_LINEAR):
    H, W = image.shape[:2]
    scale_ratio = (crop_size[0]+1) / min(H, W)
    if scale_ratio > 1.0:
        Hs, Ws = int(H*scale_ratio), int(W*scale_ratio)
        image = cv2.resize(image, (Ws,Hs), interpolation=resize_method)
        if label is not None:
            label = cv2.resize(label, (Ws,Hs), interpolation=cv2.INTER_NEAREST)


                    
@show_data_information(SHOW_PREPROCESSING, op_name='rotate')
def random_rotate(image, label=None, min_angle=0, max_angle=0, center=None, scale=1.0, borderValue=0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    # assert max_angle > min_angle
    # if min_angle == max_angle:
    #     angle = min_angle
    # else:
    #     angle = np.random.uniform(min_angle, max_angle)
    
    angle = get_random_uniform_value(min_angle, max_angle, 1)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    image = cv2.warpAffine(image, M, (w, h), borderValue)
    if label is not None:
        label = cv2.warpAffine(label, M, (w, h), borderValue)
    return (image, label)


@show_data_information(SHOW_PREPROCESSING, op_name='gaussian_blur')
def random_gaussian(image, label=None, 
    min_std=0.0, max_std=1.5, std_step_size=0.1, kernel_size=(3,3)):
    std_value = get_random_uniform_value(min_std, max_std, std_step_size)
    image = cv2.GaussianBlur(image, ksize=kernel_size, sigmaX=std_value, sigmaY=std_value)
    if label is not None:
        label = cv2.GaussianBlur(label, ksize=kernel_size, sigmaX=std_value, sigmaY=std_value)
    return (image, label)


def gaussian_blur(image, label=None, kernel_size=(7,7)):
    assert (isinstance(kernel_size, tuple) or isinstance(kernel_size, list))
    image = cv2.GaussianBlur(image, kernel_size, 0)
    if label is not None:
        label = cv2.GaussianBlur(label, kernel_size, 0)
    return (image, label)


@show_data_information(SHOW_PREPROCESSING, op_name='flip')
def random_flip(image, label=None, flip_prob=0.5, flip_mode='H'):
    def rand_flip_op(image, label, flip_mode_code):
        randnum = np.random.uniform(0.0, 1.0)
        if flip_prob > randnum:
            image = cv2.flip(image, flip_mode_code)
            if label is not None:
                label = cv2.flip(label, flip_mode_code)
        return image, label
    # Horizontal flipping
    if flip_mode in ['H', 'HV', 'VH']:
        image, label = rand_flip_op(image, label, flip_mode_code=1)
    # Vertical flipping
    if flip_mode in ['V', 'HV', 'VH']:
        image, label = rand_flip_op(image, label, flip_mode_code=0)
    # Flip in vertical and horizontal
    if flip_mode == '_HV':
        image, label = rand_flip_op(image, label, flip_mode_code=-1)

    return image, label


def HU_to_pixelvalue():
    pass


def z_score_normalize(image):
    return np.uint8((image-np.mean(image)) / np.std(image))


@show_data_information(SHOW_PREPROCESSING, op_name='crop')
def random_crop(image, label, crop_size):
    Hs, Ws = image.shape[:2]
    Ws = np.random.randint(0, Ws - crop_size[1] + 1, 1)[0]
    Hs = np.random.randint(0, Hs - crop_size[0] + 1, 1)[0]
    image = image[Hs:Hs + crop_size[0], Ws:Ws + crop_size[0]]
    if label is not None:
        label = label[Hs:Hs + crop_size[1], Ws:Ws + crop_size[1]]
    return image, label


@show_data_information(SHOW_PREPROCESSING, op_name='scale')
def random_scale(image, label, min_scale_factor, max_scale_factor, step_size, resize_method=cv2.INTER_LINEAR):
    scale = get_random_scale(min_scale_factor, max_scale_factor, step_size)
    Hs, Ws = image.shape[:2]
    Hs, Ws = int(scale*Hs), int(scale*Ws)
    image = cv2.resize(image, (Ws,Hs), interpolation=resize_method)
    if label is not None:
        label = cv2.resize(label, (Ws,Hs), interpolation=cv2.INTER_NEAREST)
    return image, label

