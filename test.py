# -*_coding:utf-8-*_
import os
from PIL import Image
import tensorflow as tf
from tensorflow.contrib import slim as slim
import numpy as np
import cv2
from preprocessing import preprocessing_factory, vgg_preprocessing

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["DYLD_PRINT_LIBRARIES"] = "1"

im = np.array(Image.open('/Users/qiuyurui/Desktop/74794046.jpg'), dtype=np.float32)
# inputs = tf.placeholder('float32', shape=[None, None, 3], name='input')
# preprocess_inputs = vgg_preprocessing.preprocess_image(inputs, 224, 224, is_training=False)
# preprocess_inputs = tf.expand_dims(preprocess_inputs, 0)
# with slim.arg_scope(resnet_v1.resnet_arg_scope()):
#     net, endpoints = resnet_v1.resnet_v1_50(preprocess_inputs, num_classes=12, is_training=False)
# # prob = tf.nn.softmax(net)
#
# sess = tf.InteractiveSession()
# variables = slim.get_variables_to_restore()
# init_fn = slim.assign_from_checkpoint_fn('/Users/qiuyurui/Desktop/car_color/model_slim/model.ckpt-110887', variables, ignore_missing_vars=True)
# init_fn(sess)


# t = sess.run(endpoints['predictions'], feed_dict={inputs: im})
# print(np.argmax(t))

# with tf.Graph().as_default():
#     tf_global_step = tf.train.get_or_create_global_step()
#     inputs = tf.placeholder('float32', shape=[None, None, 3], name='input')
#     preprocess_inputs = vgg_preprocessing.preprocess_image(inputs, 224, 224, is_training=False)
#     preprocess_inputs = tf.expand_dims(preprocess_inputs, 0, name='image')
#
#     sess_pre = tf.Session()
# sess = tf.Session()
# print(preprocess_inputs)
# op = sess_pre.graph.get_tensor_by_name('image:0')
# image = sess_pre.run(op, feed_dict={inputs: im})
# print(image)

# with tf.Graph().as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile('/Users/qiuyurui/Desktop/car_color/resnet_50_slim.pb', 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')
#     sess = tf.Session()
#     input_image_tensor = sess.graph.get_tensor_by_name("input:0")
#     output_tensor_name = sess.graph.get_tensor_by_name("resnet_v1_50/predictions/Reshape_1:0")
# out = sess.run(output_tensor_name, feed_dict={input_image_tensor: image})
# print(np.argmax(out))


class MultiClassPredict(object):
    def __init__(self, model_path, thresh, model_name):

        try:

            with tf.Graph().as_default():
                tf_global_step = tf.train.get_or_create_global_step()
                inputs = tf.placeholder('float32', shape=[None, None, 3], name='img_in')
                preprocess_inputs = preprocessing_factory.get_preprocessing(model_name)(inputs, 224, 224)
                preprocess_inputs = tf.expand_dims(preprocess_inputs, 0, name='image')

                self.sess_pre = tf.Session()

                self.pro_op = self.sess_pre.graph.get_tensor_by_name('image:0')

            with tf.Graph().as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                self.sess = tf.Session()
                # self.input_image_tensor = self.sess.graph.get_tensor_by_name("Placeholder:0")
                # self.output_tensor_name = self.sess.graph.get_tensor_by_name("final_result:0")
                # self.input_image_tensor = self.sess.graph.get_tensor_by_name("images:0")
                # self.output_tensor_name = self.sess.graph.get_tensor_by_name("InceptionV3/Predictions/Softmax:0")
                self.input_image_tensor = self.sess.graph.get_tensor_by_name("input:0")
                self.output_tensor_name = self.sess.graph.get_tensor_by_name("resnet_v1_50/predictions/Reshape_1:0")
                self.score_thresh = thresh

            print('init ColorPredict model ok')

        except Exception as e:
            print('init ColorPredict model error!', e)

    def predict(self, img_in):

        image = self.sess_pre.run(self.pro_op, feed_dict={'img_in:0': img_in})
        out = self.sess.run(self.output_tensor_name, feed_dict={self.input_image_tensor: image})
        all_scores = [round(float(i), 3) for i in out[0]]
        max_score = max(all_scores)

        return max_score, all_scores


if __name__ == "__main__":
    import glob
    import sys

    sys.path.append('/Users/qiuyurui/PycharmProjects/image_review')
    from Modules.CarBodyModule.CarBody import CarDetect

    car_model = '/Users/qiuyurui/PycharmProjects/image_review/Modules/CarBodyModule/ssd_car_20560_inference_graph.pb'
    car_detector = CarDetect(car_model, thresh=0.5)

    model_graph = '/Users/qiuyurui/Desktop/car_color/resnet_50_slim_car.pb'
    # '/Users/qiuyurui/Desktop/car_color/resnet_50_graph.pb'
    # /Users/qiuyurui/Desktop/car_color/inception_color_graph.pb car_color/legacy/
    test_pic = '/Users/qiuyurui/Desktop/car_color/legacy/'
    detector = MultiClassPredict(model_graph, 0.5, 'resnet_v1_50')
    img_files = glob.glob(test_pic + '*.jpg')
    color_dict = {0: "black", 1: "white", 2: "sliver_gray", 3: "deep_gray", 4: "coffee", 5: "red",
                  6: "blue", 7: "green", 8: "yellow", 9: "orange", 10: "champagne", 11: "purple"}
    print('---start----')
    for file in img_files:
        try:

            img_rgb = np.array(Image.open(file), dtype=np.uint8)
            car_box, car_score = car_detector.predict(img_rgb)
            img_in = img_rgb.copy()[car_box[0]:car_box[2], car_box[1]:car_box[3]]
            h, w = img_in.shape[0:2]
            # img_in = img_in[int(h / 2) - int(h / 5):h - int(h / 5), :]
            max_score, scores = detector.predict(img_in.copy())
            maxIndex = scores.index(max_score)
            color = color_dict[maxIndex]
            print(maxIndex, scores, color)
            cv2.imshow('show', cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)

        except Exception as e:
            print(e)
