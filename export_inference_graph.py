# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Saves out a GraphDef containing the architecture of the model.
To use it, run something like this, with a model name defined by slim:
bazel build tensorflow_models/research/slim:export_inference_graph
bazel-bin/tensorflow_models/research/slim/export_inference_graph \
--model_name=inception_v3 --output_file=/tmp/inception_v3_inf_graph.pb
If you then want to use the resulting model with your own or pretrained
checkpoints as part of a mobile model, you can run freeze_graph to get a graph
def with the variables inlined as constants using:
bazel build tensorflow/python/tools:freeze_graph
bazel-bin/tensorflow/python/tools/freeze_graph \
--input_graph=/tmp/inception_v3_inf_graph.pb \
--input_checkpoint=/tmp/checkpoints/inception_v3.ckpt \
--input_binary=true --output_graph=/tmp/frozen_inception_v3.pb \
--output_node_names=InceptionV3/Predictions/Reshape_1
The output node names will vary depending on the model, but you can inspect and
estimate them using the summarize_graph tool:
bazel build tensorflow/tools/graph_transforms:summarize_graph
bazel-bin/tensorflow/tools/graph_transforms/summarize_graph \
--in_graph=/tmp/inception_v3_inf_graph.pb
To run the resulting graph in C++, you can look at the label_image sample code:
bazel build tensorflow/examples/label_image:label_image
bazel-bin/tensorflow/examples/label_image/label_image \
--image=${HOME}/Pictures/flowers.jpg \
--input_layer=input \
--output_layer=InceptionV3/Predictions/Reshape_1 \
--graph=/tmp/frozen_inception_v3.pb \
--labels=/tmp/imagenet_slim_labels.txt \
--input_mean=0 \
--input_std=255
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import graph_util

from tensorflow.python.platform import gfile
from datasets import dataset_factory
from nets import nets_factory
import argparse
import re
import sys
import os
from google.protobuf import text_format

from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import saver_pb2
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.client import session
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import importer
from tensorflow.python.platform import app
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import loader
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.tools import saved_model_utils
from tensorflow.python.training import checkpoint_management
from tensorflow.python.training import saver as saver_lib

slim = tf.contrib.slim

tf.app.flags.DEFINE_boolean(
    'is_training', False,
    'Whether to save out a training-focused version of the model.')

tf.app.flags.DEFINE_integer(
    'image_size', None,
    'The image size to use, otherwise use the model default_image_size.')

tf.app.flags.DEFINE_integer(
    'batch_size', None,
    'Batch size for the exported model. Defaulted to "None" so batch size can '
    'be specified at model runtime.')

tf.app.flags.DEFINE_string('dataset_name', 'mydata',
                           'The name of the dataset to use with the model.')

tf.app.flags.DEFINE_string(
    'dataset_dir', '', 'Directory to save intermediate dataset files to')

tf.app.flags.DEFINE_string('input_checkpoint', '/Users/qiuyurui/Desktop/car_color/model_slim/model.ckpt-110887',
                           'Path to trained checkpoint, typically of the form path/to/model.ckpt')

tf.app.flags.DEFINE_string(
    'model_name', 'resnet_v1_50', 'The name of the architecture to save.')

tf.app.flags.DEFINE_integer(
    'num_classes', 12,
    'the number of the classes to predict')

tf.app.flags.DEFINE_string(
    'output_file', '/Users/qiuyurui/Desktop/car_color/resnet_50_slim_car.pb', 'Where to save the resulting file to.')

tf.app.flags.DEFINE_string(
    'scope', 'resnet_v1_50', 'the scope name of the pb model')

FLAGS = tf.app.flags.FLAGS


def _has_no_variables(sess):
    """Determines if the graph has any variables.

    Args:
      sess: TensorFlow Session.

    Returns:
      Bool.
    """
    for op in sess.graph.get_operations():
        if op.type.startswith("Variable") or op.type.endswith("VariableOp"):
            return False
    return True


def freeze_graph_with_def_protos(input_graph_def,
                                 input_saver_def,
                                 input_checkpoint,
                                 output_node_names,
                                 restore_op_name,
                                 filename_tensor_name,
                                 output_graph,
                                 clear_devices,
                                 initializer_nodes,
                                 variable_names_whitelist="",
                                 variable_names_blacklist="",
                                 input_meta_graph_def=None,
                                 input_saved_model_dir=None,
                                 saved_model_tags=None,
                                 checkpoint_version=saver_pb2.SaverDef.V2):
    """Converts all variables in a graph and checkpoint into constants.

    Args:
      input_graph_def: A `GraphDef`.
      input_saver_def: A `SaverDef` (optional).
      input_checkpoint: The prefix of a V1 or V2 checkpoint, with V2 taking
        priority.  Typically the result of `Saver.save()` or that of
        `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
        V1/V2.
      output_node_names: The name(s) of the output nodes, comma separated.
      restore_op_name: Unused.
      filename_tensor_name: Unused.
      output_graph: String where to write the frozen `GraphDef`.
      clear_devices: A Bool whether to remove device specifications.
      initializer_nodes: Comma separated string of initializer nodes to run before
                         freezing.
      variable_names_whitelist: The set of variable names to convert (optional, by
                                default, all variables are converted).
      variable_names_blacklist: The set of variable names to omit converting
                                to constants (optional).
      input_meta_graph_def: A `MetaGraphDef` (optional),
      input_saved_model_dir: Path to the dir with TensorFlow 'SavedModel' file
                             and variables (optional).
      saved_model_tags: Group of comma separated tag(s) of the MetaGraphDef to
                        load, in string format (optional).
      checkpoint_version: Tensorflow variable file format (saver_pb2.SaverDef.V1
                          or saver_pb2.SaverDef.V2)

    Returns:
      Location of the output_graph_def.
    """
    del restore_op_name, filename_tensor_name  # Unused by updated loading code.

    # 'input_checkpoint' may be a prefix if we're using Saver V2 format
    if (not input_saved_model_dir and
            not checkpoint_management.checkpoint_exists(input_checkpoint)):
        print("Input checkpoint '" + input_checkpoint + "' doesn't exist!")
        return -1

    if not output_node_names:
        print("You need to supply the name of a node to --output_node_names.")
        return -1

    # Remove all the explicit device specifications for this node. This helps to
    # make the graph more portable.
    if clear_devices:
        if input_meta_graph_def:
            for node in input_meta_graph_def.graph_def.node:
                node.device = ""
        elif input_graph_def:
            for node in input_graph_def.node:
                node.device = ""

    if input_graph_def:
        _ = importer.import_graph_def(input_graph_def, name="")
    with session.Session() as sess:
        if input_saver_def:
            saver = saver_lib.Saver(
                saver_def=input_saver_def, write_version=checkpoint_version)
            saver.restore(sess, input_checkpoint)
        elif input_meta_graph_def:
            restorer = saver_lib.import_meta_graph(
                input_meta_graph_def, clear_devices=True)
            restorer.restore(sess, input_checkpoint)
            if initializer_nodes:
                sess.run(initializer_nodes.replace(" ", "").split(","))
        elif input_saved_model_dir:
            if saved_model_tags is None:
                saved_model_tags = []
            loader.load(sess, saved_model_tags, input_saved_model_dir)
        else:
            var_list = {}
            reader = pywrap_tensorflow.NewCheckpointReader(input_checkpoint)
            var_to_shape_map = reader.get_variable_to_shape_map()

            # List of all partition variables. Because the condition is heuristic
            # based, the list could include false positives.
            all_parition_variable_names = [
                tensor.name.split(":")[0]
                for op in sess.graph.get_operations()
                for tensor in op.values()
                if re.search(r"/part_\d+/", tensor.name)
            ]
            has_partition_var = False

            for key in var_to_shape_map:
                try:
                    tensor = sess.graph.get_tensor_by_name(key + ":0")
                    if any(key in name for name in all_parition_variable_names):
                        has_partition_var = True
                except KeyError:
                    # This tensor doesn't exist in the graph (for example it's
                    # 'global_step' or a similar housekeeping element) so skip it.
                    continue
                var_list[key] = tensor

            try:
                saver = saver_lib.Saver(
                    var_list=var_list, write_version=checkpoint_version)
            except TypeError as e:
                # `var_list` is required to be a map of variable names to Variable
                # tensors. Partition variables are Identity tensors that cannot be
                # handled by Saver.
                if has_partition_var:
                    print("Models containing partition variables cannot be converted "
                          "from checkpoint files. Please pass in a SavedModel using "
                          "the flag --input_saved_model_dir.")
                    return -1
                # Models that have been frozen previously do not contain Variables.
                elif _has_no_variables(sess):
                    print("No variables were found in this model. It is likely the model "
                          "was frozen previously. You cannot freeze a graph twice.")
                    return 0
                else:
                    raise e

            saver.restore(sess, input_checkpoint)
            if initializer_nodes:
                sess.run(initializer_nodes.replace(" ", "").split(","))

        variable_names_whitelist = (
            variable_names_whitelist.replace(" ", "").split(",")
            if variable_names_whitelist else None)
        variable_names_blacklist = (
            variable_names_blacklist.replace(" ", "").split(",")
            if variable_names_blacklist else None)

        if input_meta_graph_def:
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_meta_graph_def.graph_def,
                output_node_names.replace(" ", "").split(","),
                variable_names_whitelist=variable_names_whitelist,
                variable_names_blacklist=variable_names_blacklist)
        else:
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.replace(" ", "").split(","),
                variable_names_whitelist=variable_names_whitelist,
                variable_names_blacklist=variable_names_blacklist)

    # Write GraphDef to file if output path has been given.
    if output_graph:
        with gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())

    return output_graph_def


def _parse_input_graph_proto(input_graph, input_binary):
    """Parser input tensorflow graph into GraphDef proto."""
    if not gfile.Exists(input_graph):
        print("Input graph file '" + input_graph + "' does not exist!")
        return -1
    input_graph_def = graph_pb2.GraphDef()
    mode = "rb" if input_binary else "r"
    with gfile.FastGFile(input_graph, mode) as f:
        if input_binary:
            input_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_graph_def)
    return input_graph_def


def _parse_input_meta_graph_proto(input_graph, input_binary):
    """Parser input tensorflow graph into MetaGraphDef proto."""
    if not gfile.Exists(input_graph):
        print("Input meta graph file '" + input_graph + "' does not exist!")
        return -1
    input_meta_graph_def = MetaGraphDef()
    mode = "rb" if input_binary else "r"
    with gfile.FastGFile(input_graph, mode) as f:
        if input_binary:
            input_meta_graph_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), input_meta_graph_def)
    print("Loaded meta graph file '" + input_graph)
    return input_meta_graph_def


def _parse_input_saver_proto(input_saver, input_binary):
    """Parser input tensorflow Saver into SaverDef proto."""
    if not gfile.Exists(input_saver):
        print("Input saver file '" + input_saver + "' does not exist!")
        return -1
    mode = "rb" if input_binary else "r"
    with gfile.FastGFile(input_saver, mode) as f:
        saver_def = saver_pb2.SaverDef()
        if input_binary:
            saver_def.ParseFromString(f.read())
        else:
            text_format.Merge(f.read(), saver_def)
    return saver_def


def freeze_graph(input_graph,
                 input_saver,
                 input_binary,
                 input_checkpoint,
                 output_node_names,
                 restore_op_name,
                 filename_tensor_name,
                 output_graph,
                 clear_devices,
                 initializer_nodes,
                 variable_names_whitelist="",
                 variable_names_blacklist="",
                 input_meta_graph=None,
                 input_saved_model_dir=None,
                 saved_model_tags=tag_constants.SERVING,
                 checkpoint_version=saver_pb2.SaverDef.V2):
    """Converts all variables in a graph and checkpoint into constants.

    Args:
      input_graph: A `GraphDef` file to load.
      input_saver: A TensorFlow Saver file.
      input_binary: A Bool. True means input_graph is .pb, False indicates .pbtxt.
      input_checkpoint: The prefix of a V1 or V2 checkpoint, with V2 taking
        priority.  Typically the result of `Saver.save()` or that of
        `tf.train.latest_checkpoint()`, regardless of sharded/non-sharded or
        V1/V2.
      output_node_names: The name(s) of the output nodes, comma separated.
      restore_op_name: Unused.
      filename_tensor_name: Unused.
      output_graph: String where to write the frozen `GraphDef`.
      clear_devices: A Bool whether to remove device specifications.
      initializer_nodes: Comma separated list of initializer nodes to run before
                         freezing.
      variable_names_whitelist: The set of variable names to convert (optional, by
                                default, all variables are converted),
      variable_names_blacklist: The set of variable names to omit converting
                                to constants (optional).
      input_meta_graph: A `MetaGraphDef` file to load (optional).
      input_saved_model_dir: Path to the dir with TensorFlow 'SavedModel' file and
                             variables (optional).
      saved_model_tags: Group of comma separated tag(s) of the MetaGraphDef to
                        load, in string format.
      checkpoint_version: Tensorflow variable file format (saver_pb2.SaverDef.V1
                          or saver_pb2.SaverDef.V2).
    Returns:
      String that is the location of frozen GraphDef.
    """
    input_graph_def = None
    if input_saved_model_dir:
        input_graph_def = saved_model_utils.get_meta_graph_def(
            input_saved_model_dir, saved_model_tags).graph_def
    elif input_graph:
        input_graph_def = _parse_input_graph_proto(input_graph, input_binary)
    input_meta_graph_def = None
    if input_meta_graph:
        input_meta_graph_def = _parse_input_meta_graph_proto(
            input_meta_graph, input_binary)
    input_saver_def = None
    if input_saver:
        input_saver_def = _parse_input_saver_proto(input_saver, input_binary)
    freeze_graph_with_def_protos(
        input_graph_def,
        input_saver_def,
        input_checkpoint,
        output_node_names,
        restore_op_name,
        filename_tensor_name,
        output_graph,
        clear_devices,
        initializer_nodes,
        variable_names_whitelist,
        variable_names_blacklist,
        input_meta_graph_def,
        input_saved_model_dir,
        saved_model_tags.replace(" ", "").split(","),
        checkpoint_version=checkpoint_version)


def main(_):
    if not FLAGS.output_file:
        raise ValueError('You must supply the path to save to with --output_file')
    tf.logging.set_verbosity(tf.logging.INFO)
    pb_file_path = 'pb_structure'
    middle_pb = "./{}/{}.pb".format(pb_file_path, FLAGS.model_name)
    if not os.path.exists(pb_file_path):
        os.mkdir(pb_file_path)
    with tf.Graph().as_default() as graph:
        with tf.Session() as sess:
            # dataset = dataset_factory.get_dataset(FLAGS.dataset_name, 'train',
            #                                       FLAGS.dataset_dir)
            network_fn = nets_factory.get_network_fn(
                FLAGS.model_name,
                num_classes=FLAGS.num_classes,  # (dataset.num_classes - FLAGS.labels_offset),
                is_training=FLAGS.is_training)
            image_size = FLAGS.image_size or network_fn.default_image_size
            placeholder = tf.placeholder(name='input', dtype=tf.float32,
                                         shape=[FLAGS.batch_size, image_size,
                                                image_size, 3])
            net, end_points = network_fn(placeholder)

            names = list(end_points.keys())
            print(names[-1])

            graph_def = graph.as_graph_def()
            with gfile.GFile(middle_pb, 'wb') as f:
                f.write(graph_def.SerializeToString())

    output_node_names = '{}/{}/Reshape_1'.format(FLAGS.scope, names[-1])
    freeze_graph(middle_pb, "", True,
                 FLAGS.input_checkpoint, output_node_names,
                 "save/restore_all", "save/Const:0",
                 FLAGS.output_file, True, ""
                 )


if __name__ == '__main__':
    tf.app.run()
