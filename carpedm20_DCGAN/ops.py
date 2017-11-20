import math
import numpy as np 
import tensorflow as tf
import tensorflow.contrib.layers as tfcl
from tensorflow.python.framework import ops

from utils import *


class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum=0.9, name="batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tfcl.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                           scale=True, is_training=train, scope=self.name)


class identity_op(object):
  def __init__(self, name="identity_op"):
    with tf.variable_scope(name):
      self.name = name
                    
  def __call__(self, x, train=True):
    return tf.identity(x)


def concat(tensors, axis, *args, **kwargs):
    return tf.concat(tensors, axis, *args, **kwargs)


def conv_cond_concat(x, y):
  x_shapes = x.get_shape()
  y_shapes = y.get_shape()
  return concat([x, y*tf.ones([x_shapes[0], x_shapes[1], x_shapes[2], y_shapes[3]])], 3)


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))


def conv2d(input, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d", sum=False):
  with tf.variable_scope(name):
    filters = tf.get_variable('filters', [k_h, k_w, input.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
    conv = tf.nn.conv2d(input, filters, strides=[1, d_h, d_w, 1], padding='SAME')

    biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
    
    if sum:
      tf.summary.histogram(name+"_filters", filters, family='net_vars')
      tf.summary.histogram(name+"_biases", biases, family='net_vars')

    return conv


def deconv2d(input, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d", sum=False):
  with tf.variable_scope(name):
    filters = tf.get_variable('filters', [k_h, k_w, output_shape[-1], input.get_shape()[-1]], initializer=tf.random_normal_initializer(stddev=stddev))
    
    deconv = tf.nn.conv2d_transpose(input, filters, output_shape=output_shape, strides=[1, d_h, d_w, 1])

    biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    if sum:
      tf.summary.histogram(name+"_filters", filters, family='net_vars')
      tf.summary.histogram(name+"_biases", biases, family='net_vars')
  
    return deconv


def lrelu(x, leak=0.2, name="lrelu"):
  return tf.maximum(x, leak*x)


def linear(input, output_size, name="linear", stddev=0.02, bias_start=0.0, sum=False):
  shape = input.get_shape().as_list()

  with tf.variable_scope(name):
    weights = tf.get_variable("weights", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
    biases = tf.get_variable("biases", [output_size], initializer=tf.constant_initializer(bias_start))
    
    if sum:
      tf.summary.histogram(name+"_weights", weights, family='net_vars')
      tf.summary.histogram(name+"_biases", biases, family='net_vars')
    
    return tf.matmul(input, weights) + biases
