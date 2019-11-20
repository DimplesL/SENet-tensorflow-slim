from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def inception_preprocessing(_image, crop_height, crop_width):
    _image = _image.astype(np.float32)
    _image_crop = _central_crop(_image, crop_height, crop_width)
    _image_crop = np.multiply(_image_crop, 1.0 / 255)
    _image_crop = np.subtract(_image_crop, 0.5)
    _image_crop = np.multiply(_image_crop, 2.0)
    return _image_crop


def lenet_preprocessing(_image, crop_height, crop_width):
    _image = _image.astype(np.float32)
    _image_crop = _central_crop(_image, crop_height, crop_width)
    _image_crop = np.subtract(_image_crop, 128.0)
    _image_crop = np.divide(_image_crop, 128.0)
    return _image_crop


def vgg_preprocessing(_image, crop_height, crop_width):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    if _image.shape[-1] != 3:
        raise ValueError('Input must be of size [height, width, C>0]')
    _image = _image.astype(np.float32)
    _image_crop = _central_crop(_image, crop_height, crop_width)
    means = [_R_MEAN, _G_MEAN, _B_MEAN]
    return np.subtract(_image_crop, means)


def _central_crop(image, crop_height, crop_width):
    """Performs central crops of the given image list.

    Args:
      image: the input image
      crop_height: the height of the image following the crop.
      crop_width: the width of the image following the crop.

    Returns:
      the cropped images.
    """

    image_height = image.shape[0]
    image_width = image.shape[1]

    offset_height = int((image_height - crop_height) / 2.0)
    offset_width = int((image_width - crop_width) / 2.0)

    return image[offset_height:crop_height + offset_height, offset_width:crop_width + offset_width, :]


def get_preprocessing(name):
    """Returns preprocessing_fn(image, height, width, **kwargs).

    Args:
      name: The name of the preprocessing function.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      preprocessing_fn: A function that preprocessing a single image (pre-batch).
        It has the following signature:
          image = preprocessing_fn(image, output_height, output_width, ...).

    Raises:
      ValueError: If Preprocessing `name` is not recognized.
    """
    preprocessing_fn_map = {
        'inception': inception_preprocessing,
        'inception_v1': inception_preprocessing,
        'inception_v2': inception_preprocessing,
        'inception_v3': inception_preprocessing,
        'inception_v4': inception_preprocessing,
        'inception_resnet_v2': inception_preprocessing,
        'lenet': lenet_preprocessing,
        'mobilenet_v1': inception_preprocessing,
        'nasnet_mobile': inception_preprocessing,
        'nasnet_large': inception_preprocessing,
        'pnasnet_large': inception_preprocessing,
        'resnet_v1_50': vgg_preprocessing,
        'resnet_v1_101': vgg_preprocessing,
        'resnet_v1_152': vgg_preprocessing,
        'resnet_v1_200': vgg_preprocessing,
        'resnet_v2_50': vgg_preprocessing,
        'resnet_v2_101': vgg_preprocessing,
        'resnet_v2_152': vgg_preprocessing,
        'resnet_v2_200': vgg_preprocessing,
        'vgg': vgg_preprocessing,
        'vgg_a': vgg_preprocessing,
        'vgg_16': vgg_preprocessing,
        'vgg_19': vgg_preprocessing,
    }

    if name not in preprocessing_fn_map:
        raise ValueError('Preprocessing name [%s] was not recognized' % name)

    return preprocessing_fn_map[name]
