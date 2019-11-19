import os
import tensorflow as tf
import glob
from datasets import dataset_utils
import tensorflow.contrib.slim as slim


def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    _NUM_CLASSES = len(glob.glob(dataset_dir + '/*/'))
    ALL_NUM = len(glob.glob(dataset_dir + '/*/*.jpg'))
    NUM_VAL = int(ALL_NUM * 0.05)
    SPLITS_TO_SIZES = {'train': ALL_NUM - NUM_VAL, 'validation': NUM_VAL}

    _FILE_PATTERN = 'mydata_%s_*.tfrecord'

    _ITEMS_TO_DESCRIPTIONS = {
        'image': 'A color image of varying size.',
        'label': 'A single integer between 0 and 257',
    }

    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature([], tf.int64, default_value=tf.zeros([], dtype=tf.int64))
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label')
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    labels_to_names = None
    if dataset_utils.has_labels(dataset_dir):
        labels_to_names = dataset_utils.read_label_file(dataset_dir)

    return slim.dataset.Dataset(data_sources=file_pattern,
                                reader=reader,
                                decoder=decoder,
                                num_samples=SPLITS_TO_SIZES[split_name],
                                items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
                                num_classes=_NUM_CLASSES,
                                labels_to_names=labels_to_names)
