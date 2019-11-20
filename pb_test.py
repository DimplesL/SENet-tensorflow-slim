# -*_coding:utf-8-*_
import tensorflow as tf
import cv2
import numpy as np
import os
from preprocessing import np_preprocess

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["DYLD_PRINT_LIBRARIES"] = "1"


class MultiClassPredict(object):
    def __init__(self, model_path, thresh, model_name):

        try:
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                # load model
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(model_path, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
                config = tf.ConfigProto()
                self.sess = tf.Session(config=config)
                # self.input_image_tensor = self.sess.graph.get_tensor_by_name("Placeholder:0")
                # self.output_tensor_name = self.sess.graph.get_tensor_by_name("final_result:0")
                # self.input_image_tensor = self.sess.graph.get_tensor_by_name("images:0")
                # self.output_tensor_name = self.sess.graph.get_tensor_by_name("InceptionV3/Predictions/Softmax:0")
                self.input_image_tensor = self.sess.graph.get_tensor_by_name("input:0")
                self.output_tensor_name = self.sess.graph.get_tensor_by_name("resnet_v1_50/predictions/Softmax:0")
                self.preprocess = np_preprocess.get_preprocessing(model_name)
                self.score_thresh = thresh
                # "final_result:0" shape=(?, 12)  "Placeholder:0" shape=(?, 224, 224, 3)
                # "images:0"  "InceptionV3/Predictions/Softmax:0"
                # resnet_v1_50/predictions/Softmax

            print('init ColorPredict model ok')

        except Exception as e:
            print('init ColorPredict model error!', e)

    def preprocess2(self, imgin):
        scale = 1 / 255.0
        imgpro = cv2.resize(imgin, (224, 224))
        imgpro = imgpro.astype(np.float32)
        imgpro = np.multiply(imgpro, scale)
        return imgpro

    def pre_process(self, imgin):
        # imgin = imgin.astype(np.float32)
        # imgpro = cv2.resize(imgin, (224, 224))
        imgpro = self.preprocess(imgin, 224, 224)
        return imgpro

    def predict(self, img_in):

        h, w = img_in.shape[0:2]
        img_pro = self.pre_process(img_in)
        # std_image = tf.image.per_image_standardization(img_pro)
        # std_image = tf.image.convert_image_dtype(img_pro, tf.float32)
        #
        # with tf.Session() as sess:
        #     image = sess.run(std_image)
        #     cv2.imshow('s', image)
        #     cv2.waitKey(0)

        # file_reader = tf.read_file(file)
        #
        # image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")

        # float_caster = tf.cast(img_in, tf.float32)
        # dims_expander = tf.expand_dims(float_caster, 0)
        # resized = tf.image.resize_bilinear(dims_expander, [224, 224])
        # normalized = tf.divide(tf.subtract(resized, [0]), [255])
        # sess = tf.compat.v1.Session()
        # img_pro = sess.run(normalized)
        # cv2.imshow('show', cv2.cvtColor(img_pro[0], cv2.COLOR_RGB2BGR))  # cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR)
        # cv2.waitKey(0)

        out = self.sess.run(self.output_tensor_name,
                            feed_dict={self.input_image_tensor: np.expand_dims(img_pro, axis=0)})
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

    model_graph = '/Users/qiuyurui/Desktop/car_color/resnet_50_slim.pb'
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

            # file_reader = tf.read_file(file)
            # image_reader = tf.image.decode_jpeg(file_reader, channels=3, name="jpeg_reader")
            # float_caster = tf.cast(image_reader, tf.float32)
            # dims_expander = tf.expand_dims(float_caster, 0)
            # resized = tf.image.resize_bilinear(dims_expander, [224, 224])
            # normalized = tf.divide(tf.subtract(resized, [0]), [255])
            # sess = tf.compat.v1.Session()
            # result = sess.run(normalized)
            # cv2.imshow('show', cv2.cvtColor(result[0], cv2.COLOR_RGB2BGR))
            # cv2.waitKey(0)

            img = cv2.imread(file)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            car_box, car_score = car_detector.predict(img_rgb)
            img_in = img_rgb.copy()[car_box[0]:car_box[2], car_box[1]:car_box[3]]
            h, w = img_in.shape[0:2]
            img_in = img_in[int(h / 2) - int(h / 5):h - int(h / 5), :]
            max_score, scores = detector.predict(img_in)
            maxIndex = scores.index(max_score)
            color = color_dict[maxIndex]
            print(maxIndex, scores, color)
            cv2.imshow('show', cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR))  # cv2.cvtColor(img_in, cv2.COLOR_RGB2BGR)
            cv2.waitKey(0)
        except Exception as e:
            print(e)
