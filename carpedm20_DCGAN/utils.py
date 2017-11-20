#
#From https://github.com/Newmu/dcgan_code
#
from __future__ import division
from six.moves import xrange
from time import gmtime, strftime
import scipy.misc
import numpy as np
import pickle
import pprint
import math
import random
import json
import os

import tensorflow as tf
import tensorflow.contrib.slim as slim


pp = pprint.PrettyPrinter()


def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def unpickle(file):
  with open(file, 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict


def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width,
                   resize_height, resize_width, crop)


def imread(path, grayscale = False):
  if (grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)


# RGB/RGBA 8bit=[0,255] -> [-1,1]
def transform(image, input_height, input_width,
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(image, input_height, input_width,
                                resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.


def center_crop(x, input_h, input_w, resize_h=64, resize_w=64):
  if input_w is None:
    input_w = input_h
  h, w = x.shape[:2]
  j = int(round((h - input_h)/2.))
  i = int(round((w - input_w)/2.))
  return scipy.misc.imresize(x[j:j+input_h, i:i+input_w], [resize_h, resize_w])


def save_images(images, size, dataset, path):
  return imsave(inverse_transform(images, dataset), size, path)


def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image.astype(np.uint8))


# RGB/RGBA: [-1,1] -> [0,255], MNIST: [0,1] -> [0,255]
def inverse_transform(images, dataset):
  if dataset == "mnist":
    return images*255.
  else:
    return (images+1.)*127.5 # (images+1.)/2


def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')


