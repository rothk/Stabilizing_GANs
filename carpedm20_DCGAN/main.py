#
# Based on https://github.com/carpedm20/DCGAN-tensorflow
#
# Modifications: Added Regularizer for GANs (https://arxiv.org/abs/1705.09367)
#                Various syntax modifications (preserving semantics)

from model import DCGAN
from utils import pp, show_all_variables

import tensorflow as tf
import numpy as np
import time
import json
import os

flags = tf.app.flags
flags.DEFINE_string( "dataset", "celebA", "name of the dataset [celebA, cifar10]")
flags.DEFINE_integer("epochs", 10, "epochs to train [50]")
flags.DEFINE_float(  "gamma", 0.1, "noise variance for regularizer [0.1]")
flags.DEFINE_boolean("annealing", False, "annealing gamma_0 to decay_factor*gamma_0 [False]")
flags.DEFINE_float(  "decay_factor", 0.01, "exponential annealing decay factor [0.01]")
flags.DEFINE_boolean("unreg", False, "turn regularization off.")
flags.DEFINE_boolean("rmsprop", False, "RMSProp optimizer (Adam by default).")
flags.DEFINE_boolean("gaussian_prior", False, "Gaussian prior [uniform by default]")
flags.DEFINE_float(  "disc_learning_rate", 0.0002, "(initial) learning rate.")
flags.DEFINE_float(  "gen_learning_rate", 0.0002, "(initial) learning rate.")
flags.DEFINE_integer("disc_update_steps", 1, "discriminator update steps.")
flags.DEFINE_integer("dataset_size", np.inf, "reduces total size of dataset (mainly for debugging) [np.inf]")
flags.DEFINE_integer("batch_size", 64, "batch size [64]")
flags.DEFINE_integer("input_height", 108, "image height (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "image width (will be center cropped). If None, same as input_height [None]")
flags.DEFINE_integer("output_height", 64, "output height [64]")
flags.DEFINE_integer("output_width", None, "output width. If None, same as output_height [None]")
flags.DEFINE_string( "input_fname_pattern", "*.jpg", "glob filename pattern [*]")
flags.DEFINE_string( "root_dir", "RUN_STATS", "root directory [RUN_STATS]")
flags.DEFINE_string( "checkpoint_dir", "None", "directory to load the checkpoints from [None]")
FLAGS = flags.FLAGS


def main(_):
  
  if FLAGS.input_width is None:
      FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
      FLAGS.output_width = FLAGS.output_height

  file_name = time.strftime("%Y_%m_%d_%H%M", time.localtime())
  if FLAGS.unreg:
    file_name += "_unreg_dcgan"
  else:
    file_name += "_regularized_dcgan_"+str(FLAGS.gamma)+"gamma"
  if FLAGS.annealing:
    file_name += "_annealing_"+str(FLAGS.decay_factor)+"decayfactor"
  if FLAGS.rmsprop:
    file_name += "_rmsprop"
  else:
    file_name += "_adam"
  file_name += "_"+str(FLAGS.disc_update_steps)+"dsteps"
  file_name += "_"+str(FLAGS.disc_learning_rate)+"dlnr"
  file_name += "_"+str(FLAGS.gen_learning_rate)+"glnr"
  file_name += "_"+str(FLAGS.epochs)+"epochs"
  file_name += "_"+str(FLAGS.dataset)

  log_dir = os.path.abspath(os.path.join(FLAGS.root_dir, file_name))

  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
                                

  pp.pprint(flags.FLAGS.__flags)
      


  with tf.Session() as sess:
    
    dcgan = DCGAN(sess, log_dir, FLAGS)
    
    show_all_variables()

    if FLAGS.checkpoint_dir is not "None":
      if not dcgan.load_ckpt()[0]:
        raise Exception("[!] ERROR: provide valid checkpoint_dir")
    else:
      starttime = time.time()
  
      dcgan.train(FLAGS)
  
      endtime = time.time()
      print('Total Train Time: {:.2f}'.format(endtime-starttime))

    dcgan.generate(FLAGS, option=1)


  file = open(os.path.join(log_dir, "flags.json"), 'a')
  json.dump(vars(FLAGS), file)
  file.close()




if __name__ == '__main__':
  tf.app.run()
