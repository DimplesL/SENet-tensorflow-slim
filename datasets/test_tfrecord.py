import PIL.Image
import cv2
import numpy as np
import tensorflow as tf
flags = tf.app.flags
flags.DEFINE_string('tfrecord_path', '/Users/qiuyurui/Desktop/car_color/mydata_validation_00010-of-00012.tfrecord', 'path to tfrecord file')
flags.DEFINE_string('dataset_num', '100', 'data num in dataset')
FLAGS = flags.FLAGS


def read_tfrecord(tfrecord_path, num_samples=3, num_classes=1):
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'image/class/label': tf.VarLenFeature(tf.int64),
        'image/height': tf.FixedLenFeature((), tf.int64, default_value=1),
        'image/width': tf.FixedLenFeature((), tf.int64, default_value=1)

    }

    items_to_handlers = {
        'image': tf.contrib.slim.tfexample_decoder.Image(image_key='image/encoded', format_key='image/format', channels=3),
        'height': tf.contrib.slim.tfexample_decoder.Tensor('image/height'),
        'width': tf.contrib.slim.tfexample_decoder.Tensor('image/width'),
        'label': tf.contrib.slim.tfexample_decoder.Tensor('image/class/label'),
    }

    decoder = tf.contrib.slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = tf.contrib.slim.dataset.Dataset(data_sources=tfrecord_path,
                                              reader=tf.TFRecordReader,
                                              decoder=decoder,
                                              num_samples=num_samples,
                                              items_to_descriptions=None,
                                              num_classes=num_classes,)

    provider = tf.contrib.slim.dataset_data_provider.DatasetDataProvider(dataset=dataset,
                                                                         num_readers=1,
                                                                         shuffle=False,
                                                                         common_queue_capacity=256,
                                                                         common_queue_min=128,
                                                                         seed=None)

    image, height, width, label = provider.get(['image', 'height', 'width', 'label'])
    return image, height, width, label


def show_data(tf_image, tf_height, tf_width, tf_label, num):
    cv2.namedWindow('pic', 0)
    cv2.resizeWindow('pic', 2000, 1500)

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(num):
            print("______________________image({})___________________".format(i))
            image, height, width, label = sess.run([tf_image, tf_height, tf_width, tf_label])
            print("image shape is: ", image.shape)

            print("image width is: ", width)
            print("image height is: ", height)

            print("label is: ", label)

            image = tf.reshape(image, [height, width, 3])
            image_pil = PIL.Image.fromarray(image.eval(), 'RGB')
            image_cv2 = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

            cv2.imshow('pic', image_cv2)
            cv2.waitKey(0)

    coord.request_stop()
    coord.join(threads)


def main():
    image, height, width, label = read_tfrecord(FLAGS.tfrecord_path)
    show_data(image, height, width, label, int(FLAGS.dataset_num))


if __name__ == '__main__':
    main()
