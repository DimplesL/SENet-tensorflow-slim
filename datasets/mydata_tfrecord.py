import tensorflow as tf
import os
import random
import math
import sys
import argparse

_RANDOM_SEED = 0

from datasets import dataset_utils


class ImageReader(object):
    def __init__(self):
        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        # convert image data to uint8 tenser
        self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg, feed_dict={self._decode_jpeg_data: image_data})
        assert (len(image.shape) == 3)
        assert (image.shape[2] == 3)
        return image


def _get_dataset_filename(dataset_dir, split_name, shard_id, _NUM_SHARDS):
    output_filename = 'mydata_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _dataset_exists(dataset_dir, _NUM_SHARDS):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id, _NUM_SHARDS)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def _get_filenames_and_classes(dataset_dir):
    mydata_root = dataset_dir
    directories = []
    class_names = []
    for filename in os.listdir(mydata_root):
        path = os.path.join(mydata_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)
    _NUM_SHARDS = len(class_names)
    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names), _NUM_SHARDS


def _clean_up_temporary_files(dataset_dir):
    for file in os.listdir(dataset_dir):
        path = os.path.join(dataset_dir, file)
        if os.path.isdir(path):
            tf.gfile.DeleteRecursively(path)


def _convert_dataset(split_name, filename, class_names_to_ids, dataset_dir, _NUM_SHARDS):
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filename) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()
        with tf.Session('') as sess:
            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(dataset_dir, split_name, shard_id, _NUM_SHARDS)
                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id + 1) * num_per_shard, len(filename))
                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i + 1, len(filename), shard_id))
                        sys.stdout.flush()
                        try:
                            image_data = tf.gfile.FastGFile(filename[i], 'rb').read()
                            height, width = image_reader.read_image_dims(sess, image_data)

                            class_name = os.path.basename(os.path.dirname(filename[i]))
                            class_id = class_names_to_ids[class_name]

                            example = dataset_utils.image_to_tfexample(image_data, b'jpg', height, width, class_id)
                            tfrecord_writer.write(example.SerializeToString())
                        except Exception as e:
                            print(e)


def main(dataset_dir, _RATIO_VALIDATION):
    if not tf.gfile.Exists(dataset_dir):
        tf.gfile.MakeDirs(dataset_dir)

    photo_filenames, class_names, _NUM_SHARDS = _get_filenames_and_classes(dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    _NUM_VAL = int(_RATIO_VALIDATION * len(photo_filenames))
    _NUM_TRAIN = len(photo_filenames) - _NUM_VAL
    training_filenames = photo_filenames[_NUM_VAL:]
    validation_filenames = photo_filenames[:_NUM_VAL]

    _convert_dataset('train', training_filenames, class_names_to_ids, dataset_dir, _NUM_SHARDS)
    _convert_dataset('validation', validation_filenames, class_names_to_ids, dataset_dir, _NUM_SHARDS)

    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

    # _clean_up_temporary_files(dataset_dir)
    print('\n Finished converting the mydata dataset')


def get_arguments():
    parser_online = argparse.ArgumentParser()
    parser_online.add_argument('-i', '--path_input', type=str, help='origin data directory',
                               default="'/home/vip/qyr/data/car_color_data/train_new_crop/'")
    parser_online.add_argument('-e', '--eval_num', type=float, help='number of data to eval', default=0.1)

    return parser_online.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    _RATIO_VALIDATION = args.eval_num
    dataset_dir = args.path_input
    main(dataset_dir, _RATIO_VALIDATION)
