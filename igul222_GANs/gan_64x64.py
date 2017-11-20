#
# Based on https://github.com/igul222/improved_wgan_training
#
# Modifications: Added Regularizer for GANs (https://arxiv.org/abs/1705.09367)
#                Added command line parser
#                Added load_lsun.py & load_celebA.py to tflib/
#
#                Most importantly: requires python3 now! (various slight syntax modifications)

import os, sys
import time
import functools
import numpy as np
from six.moves import xrange

import tensorflow as tf
from tensorflow.python.platform import flags

import tflib as lib
import tflib.load_lsun
import tflib.load_celebA
import tflib.small_imagenet
import tflib.save_images
import tflib.plot
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.ops.layernorm

flags.DEFINE_string( "mode", "regularized_gan", "MODE: regularized_gan, gan, wgan, wgan-gp, lsgan")
flags.DEFINE_string( "architecture", "None", "choice of architecture - see GeneratorAndDiscriminator()")
flags.DEFINE_string( "dataset", "celebA", "name of the dataset [ImageNet, lsun, celebA]")
flags.DEFINE_integer("n_gpus", 1, "number of gpus to use")
flags.DEFINE_integer("iters", 100000, "How many iterations to train for")
flags.DEFINE_float(  "gamma", 0.1, "noise variance for regularizer [0.1]")
flags.DEFINE_boolean("annealing", False, "annealing gamma_0 to decay_factor*gamma_0 [False]")
flags.DEFINE_float(  "decay_factor", 0.01, "exponential annealing decay rate [0.01]")
flags.DEFINE_boolean("unreg", False, "turn regularization off when in regularized_gan mode.")
flags.DEFINE_float(  "disc_learning_rate", 0.0002, "(initial) learning rate.")
flags.DEFINE_float(  "gen_learning_rate", 0.0002, "(initial) learning rate.")
flags.DEFINE_integer("disc_update_steps", 1, "discriminator update steps.")
flags.DEFINE_integer("batch_size", 64, "batch size [64]")
flags.DEFINE_string( "root_dir", "RUN_STATS", "root directory [RUN_STATS]")
flags.DEFINE_string( "checkpoint_dir", "None", "directory to load the checkpoints from [None]")
FLAGS = flags.FLAGS


def main():

    # Download datasets and fill in the path to the extracted files here:
    # check load_*.py in tflib for the train/val folder structure.
    if FLAGS.dataset == 'lsun':
        DATA_DIR = '/local/home/rothk/data/lsun'
    elif FLAGS.dataset == 'celebA':
        DATA_DIR = '/local/home/rothk/data/celebA'
    elif FLAGS.dataset == 'ImageNet':
        DATA_DIR = '/local/home/rothk/data/ImageNet'
    else:
        raise Exception('Please specify path to data directory manually!')

    MODE = str(FLAGS.mode) # regularized_gan, gan, wgan, wgan-gp, lsgan
    N_GPUS = int(FLAGS.n_gpus) # Number of GPUs
    BATCH_SIZE = int(FLAGS.batch_size) # Must be a multiple of N_GPUS
    ITERS = int(FLAGS.iters) # How many iterations to train for
    DIM = 64 # Model dimensionality
    OUTPUT_DIM = 64*64*3 # Number of pixels in each image
    CRITIC_ITERS = 5 # How many iterations to train the critic for
    LAMBDA = 10 # Gradient penalty lambda hyperparameter

    print('\nMODE: {}, ARCHITECTURE: {}\n'.format(FLAGS.mode, FLAGS.architecture))
    lib.print_model_settings(locals().copy())


    file_name = time.strftime("%Y_%m_%d_%H%M", time.localtime())
    file_name += "_"+str(FLAGS.mode)
    file_name += "_"+str(FLAGS.architecture)
    if FLAGS.mode == "regularized_gan":
        if FLAGS.unreg:
            file_name += "_unreg"
        else:
            file_name += "_"+str(FLAGS.gamma)+"gamma"
        if FLAGS.annealing:
            file_name += "_annealing_"+str(FLAGS.decay_factor)+"decayfactor"
        file_name += "_"+str(FLAGS.disc_update_steps)+"dsteps"
        file_name += "_"+str(FLAGS.disc_learning_rate)+"dlnr"
        file_name += "_"+str(FLAGS.gen_learning_rate)+"glnr"
    file_name += "_"+str(FLAGS.iters)+"iters"
    file_name += "_"+str(FLAGS.dataset)

    log_dir = os.path.abspath(os.path.join(FLAGS.root_dir, file_name))


    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if not os.path.exists(os.path.join(log_dir, 'samples')):
        os.makedirs(os.path.join(log_dir, 'samples'))

    checkpoint_dir = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    def GeneratorAndDiscriminator():
        """
        Choose which generator and discriminator architecture to use
        """

        # ResNet Generator and Discriminator
        if FLAGS.architecture == "ResNet":
            return ResNetGenerator, ResNetDiscriminator

        # DCGAN (G: DCGAN, D: DCGAN)
        elif FLAGS.architecture == "DCGAN":
            return DCGANGenerator, DCGANDiscriminator

        # No BN and constant number of filts in G
        elif FLAGS.architecture == "ConstNumofFilters":
            return ConstNumofFiltersDCGANGenerator, DCGANDiscriminator

        # 512-dim 4-layer ReLU MLP G
        elif FLAGS.architecture == "MLP":
            return FCGenerator, DCGANDiscriminator

        # No normalization at all
        elif FLAGS.architecture == "NoNormalization":
            return functools.partial(DCGANGenerator, bn=False), functools.partial(DCGANDiscriminator, bn=False)

        # Gated multiplicative nonlinearities
        elif FLAGS.architecture == "Gated":
            return MultiplicativeDCGANGenerator, MultiplicativeDCGANDiscriminator

        # tanh nonlinearities
        elif FLAGS.architecture == "tanh":
            return functools.partial(DCGANGenerator, bn=True, nonlinearity=tf.tanh), functools.partial(DCGANDiscriminator, bn=True, nonlinearity=tf.tanh)

        # 101-layer ResNet G and D
        elif FLAGS.architecture == "ResNet101":
            return Resnet101Generator, Resnet101Discriminator

        else:
            raise Exception('You must choose an architecture!')


    # -----------------------------------------------------------------------------------
    #     JS-Regularizer
    # -----------------------------------------------------------------------------------
    def Discriminator_Regularizer(D1_logits, D1_arg, D2_logits, D2_arg):
        with tf.name_scope('disc_reg'):
            D1 = tf.nn.sigmoid(D1_logits)
            D2 = tf.nn.sigmoid(D2_logits)
            grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
            grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
            grad_D1_logits_norm = tf.norm( tf.reshape(grad_D1_logits, [BATCH_SIZE//len(DEVICES),-1]) , axis=1)
            grad_D2_logits_norm = tf.norm( tf.reshape(grad_D2_logits, [BATCH_SIZE//len(DEVICES),-1]) , axis=1)
            
            #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
            print('grad_D1_logits_norm.shape {} != D1.shape {}'.format(grad_D1_logits_norm.shape, D1.shape))
            print('grad_D2_logits_norm.shape {} != D2.shape {}'.format(grad_D2_logits_norm.shape, D2.shape))
            assert grad_D1_logits_norm.shape == D1.shape
            assert grad_D2_logits_norm.shape == D2.shape
            
            reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
            reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
            
            disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
            
            return disc_regularizer


    def LeakyReLU(x, alpha=0.2):
        return tf.maximum(alpha*x, x)

    def ReLULayer(name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
        return tf.nn.relu(output)

    def LeakyReLULayer(name, n_in, n_out, inputs):
        output = lib.ops.linear.Linear(name+'.Linear', n_in, n_out, inputs, initialization='he')
        return LeakyReLU(output)

    def Normalize(name, axes, inputs):
        if ('Discriminator' in name) and (MODE == 'wgan-gp'):
            if axes != [0,2,3]:
                raise Exception('Layernorm over non-standard axes is unsupported')
            return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs)
        elif ('Discriminator' in name) and (MODE == 'regularized_gan') and (float(str(tf.__version__)[:3]) < 1.4):
            return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=False)
        else:
            return lib.ops.batchnorm.Batchnorm(name,axes,inputs,fused=True)

    def pixcnn_gated_nonlinearity(a, b):
        return tf.sigmoid(a) * tf.tanh(b)

    def SubpixelConv2D(*args, **kwargs):
        kwargs['output_dim'] = 4*kwargs['output_dim']
        output = lib.ops.conv2d.Conv2D(*args, **kwargs)
        output = tf.transpose(output, [0,2,3,1])
        output = tf.depth_to_space(output, 2)
        output = tf.transpose(output, [0,3,1,2])
        return output

    def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
        output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
        output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
        return output

    def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
        output = inputs
        output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
        output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
        return output

    def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
        output = inputs
        output = tf.concat([output, output, output, output], axis=1)
        output = tf.transpose(output, [0,2,3,1])
        output = tf.depth_to_space(output, 2)
        output = tf.transpose(output, [0,3,1,2])
        output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
        return output

    def BottleneckResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
        """
        resample: None, 'down', or 'up'
        """
        if resample=='down':
            conv_shortcut = functools.partial(lib.ops.conv2d.Conv2D, stride=2)
            conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim//2)
            conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2, output_dim=output_dim//2, stride=2)
            conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim//2, output_dim=output_dim)
        elif resample=='up':
            conv_shortcut = SubpixelConv2D
            conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim//2)
            conv_1b       = functools.partial(lib.ops.deconv2d.Deconv2D, input_dim=input_dim//2, output_dim=output_dim//2)
            conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim//2, output_dim=output_dim)
        elif resample==None:
            conv_shortcut = lib.ops.conv2d.Conv2D
            conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim//2)
            conv_1b       = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2,  output_dim=output_dim//2)
            conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim//2, output_dim=output_dim)

        else:
            raise Exception('invalid resample value')

        if output_dim==input_dim and resample==None:
            shortcut = inputs # Identity skip-connection
        else:
            shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                     he_init=False, biases=True, inputs=inputs)

        output = inputs
        output = tf.nn.relu(output)
        output = conv_1(name+'.Conv1', filter_size=1, inputs=output, he_init=he_init)
        output = tf.nn.relu(output)
        output = conv_1b(name+'.Conv1B', filter_size=filter_size, inputs=output, he_init=he_init)
        output = tf.nn.relu(output)
        output = conv_2(name+'.Conv2', filter_size=1, inputs=output, he_init=he_init, biases=False)
        output = Normalize(name+'.BN', [0,2,3], output)

        return shortcut + (0.3*output)

    def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, he_init=True):
        """
        resample: None, 'down', or 'up'
        """
        if resample=='down':
            conv_shortcut = MeanPoolConv
            conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
            conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        elif resample=='up':
            conv_shortcut = UpsampleConv
            conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
            conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
        elif resample==None:
            conv_shortcut = lib.ops.conv2d.Conv2D
            conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim,  output_dim=input_dim)
            conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        else:
            raise Exception('invalid resample value')

        if output_dim==input_dim and resample==None:
            shortcut = inputs # Identity skip-connection
        else:
            shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1,
                                     he_init=False, biases=True, inputs=inputs)

        output = inputs
        output = Normalize(name+'.BN1', [0,2,3], output)
        output = tf.nn.relu(output)
        output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output, he_init=he_init, biases=False)
        output = Normalize(name+'.BN2', [0,2,3], output)
        output = tf.nn.relu(output)
        output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output, he_init=he_init)

        return shortcut + output


    # ! Generators

    def ResNetGenerator(n_samples, noise=None, dim=DIM, nonlinearity=tf.nn.relu):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
        output = tf.reshape(output, [-1, 8*dim, 4, 4])

        output = ResidualBlock('Generator.Res1', 8*dim, 8*dim, 3, output, resample='up')
        output = ResidualBlock('Generator.Res2', 8*dim, 4*dim, 3, output, resample='up')
        output = ResidualBlock('Generator.Res3', 4*dim, 2*dim, 3, output, resample='up')
        output = ResidualBlock('Generator.Res4', 2*dim, 1*dim, 3, output, resample='up')

        output = Normalize('Generator.OutputN', [0,2,3], output)
        output = tf.nn.relu(output)
        output = lib.ops.conv2d.Conv2D('Generator.Output', 1*dim, 3, 3, output)
        output = tf.tanh(output)

        return tf.reshape(output, [-1, OUTPUT_DIM])

    def FCGenerator(n_samples, noise=None, FC_DIM=512):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = ReLULayer('Generator.1', 128, FC_DIM, noise)
        output = ReLULayer('Generator.2', FC_DIM, FC_DIM, output)
        output = ReLULayer('Generator.3', FC_DIM, FC_DIM, output)
        output = ReLULayer('Generator.4', FC_DIM, FC_DIM, output)
        output = lib.ops.linear.Linear('Generator.Out', FC_DIM, OUTPUT_DIM, output)

        output = tf.tanh(output)

        return output

    def DCGANGenerator(n_samples, noise=None, dim=DIM, bn=True, nonlinearity=tf.nn.relu):
        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
        output = tf.reshape(output, [-1, 8*dim, 4, 4])
        if bn:
            output = Normalize('Generator.BN1', [0,2,3], output)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim, 5, output)
        if bn:
            output = Normalize('Generator.BN2', [0,2,3], output)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim, 5, output)
        if bn:
            output = Normalize('Generator.BN3', [0,2,3], output)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim, 5, output)
        if bn:
            output = Normalize('Generator.BN4', [0,2,3], output)
        output = nonlinearity(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
        output = tf.tanh(output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1, OUTPUT_DIM])

    def ConstNumofFiltersDCGANGenerator(n_samples, noise=None, dim=DIM):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*dim, noise)
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, dim, 4, 4])

        output = lib.ops.deconv2d.Deconv2D('Generator.2', dim, dim, 5, output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.3', dim, dim, 5, output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.4', dim, dim, 5, output)
        output = tf.nn.relu(output)

        output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
        output = tf.tanh(output)

        return tf.reshape(output, [-1, OUTPUT_DIM])

    def Resnet101Generator(n_samples, noise=None, dim=DIM):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim, noise)
        output = tf.reshape(output, [-1, 8*dim, 4, 4])

        for i in xrange(6):
            output = BottleneckResidualBlock('Generator.4x4_{}'.format(i), 8*dim, 8*dim, 3, output, resample=None)
        output = BottleneckResidualBlock('Generator.Up1', 8*dim, 4*dim, 3, output, resample='up')
        for i in xrange(6):
            output = BottleneckResidualBlock('Generator.8x8_{}'.format(i), 4*dim, 4*dim, 3, output, resample=None)
        output = BottleneckResidualBlock('Generator.Up2', 4*dim, 2*dim, 3, output, resample='up')
        for i in xrange(6):
            output = BottleneckResidualBlock('Generator.16x16_{}'.format(i), 2*dim, 2*dim, 3, output, resample=None)
        output = BottleneckResidualBlock('Generator.Up3', 2*dim, 1*dim, 3, output, resample='up')
        for i in xrange(6):
            output = BottleneckResidualBlock('Generator.32x32_{}'.format(i), 1*dim, 1*dim, 3, output, resample=None)
        output = BottleneckResidualBlock('Generator.Up4', 1*dim, dim//2, 3, output, resample='up')
        for i in xrange(5):
            output = BottleneckResidualBlock('Generator.64x64_{}'.format(i), dim//2, dim//2, 3, output, resample=None)

        output = lib.ops.conv2d.Conv2D('Generator.Out', dim//2, 3, 1, output, he_init=False)
        output = tf.tanh(output / 5.)

        return tf.reshape(output, [-1, OUTPUT_DIM])


    def MultiplicativeDCGANGenerator(n_samples, noise=None, dim=DIM, bn=True):
        if noise is None:
            noise = tf.random_normal([n_samples, 128])

        output = lib.ops.linear.Linear('Generator.Input', 128, 4*4*8*dim*2, noise)
        output = tf.reshape(output, [-1, 8*dim*2, 4, 4])
        if bn:
            output = Normalize('Generator.BN1', [0,2,3], output)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.deconv2d.Deconv2D('Generator.2', 8*dim, 4*dim*2, 5, output)
        if bn:
            output = Normalize('Generator.BN2', [0,2,3], output)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.deconv2d.Deconv2D('Generator.3', 4*dim, 2*dim*2, 5, output)
        if bn:
            output = Normalize('Generator.BN3', [0,2,3], output)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.deconv2d.Deconv2D('Generator.4', 2*dim, dim*2, 5, output)
        if bn:
            output = Normalize('Generator.BN4', [0,2,3], output)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.deconv2d.Deconv2D('Generator.5', dim, 3, 5, output)
        output = tf.tanh(output)

        return tf.reshape(output, [-1, OUTPUT_DIM])

    # ! Discriminators

    def ResNetDiscriminator(inputs, dim=DIM):
        output = tf.reshape(inputs, [-1, 3, 64, 64])
        output = lib.ops.conv2d.Conv2D('Discriminator.Input', 3, dim, 3, output, he_init=False)

        output = ResidualBlock('Discriminator.Res1', dim, 2*dim, 3, output, resample='down')
        output = ResidualBlock('Discriminator.Res2', 2*dim, 4*dim, 3, output, resample='down')
        output = ResidualBlock('Discriminator.Res3', 4*dim, 8*dim, 3, output, resample='down')
        output = ResidualBlock('Discriminator.Res4', 8*dim, 8*dim, 3, output, resample='down')

        output = tf.reshape(output, [-1, 4*4*8*dim])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

        return tf.reshape(output, [-1])

    def MultiplicativeDCGANDiscriminator(inputs, dim=DIM, bn=True):
        output = tf.reshape(inputs, [-1, 3, 64, 64])

        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim*2, 5, output, stride=2)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim*2, 5, output, stride=2)
        if bn:
            output = Normalize('Discriminator.BN2', [0,2,3], output)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim*2, 5, output, stride=2)
        if bn:
            output = Normalize('Discriminator.BN3', [0,2,3], output)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim*2, 5, output, stride=2)
        if bn:
            output = Normalize('Discriminator.BN4', [0,2,3], output)
        output = pixcnn_gated_nonlinearity(output[:,::2], output[:,1::2])

        output = tf.reshape(output, [-1, 4*4*8*dim])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

        return tf.reshape(output, [-1])


    def Resnet101Discriminator(inputs, dim=DIM):
        output = tf.reshape(inputs, [-1, 3, 64, 64])
        output = lib.ops.conv2d.Conv2D('Discriminator.In', 3, dim//2, 1, output, he_init=False)

        for i in xrange(5):
            output = BottleneckResidualBlock('Discriminator.64x64_{}'.format(i), dim//2, dim//2, 3, output, resample=None)
        output = BottleneckResidualBlock('Discriminator.Down1', dim//2, dim*1, 3, output, resample='down')
        for i in xrange(6):
            output = BottleneckResidualBlock('Discriminator.32x32_{}'.format(i), dim*1, dim*1, 3, output, resample=None)
        output = BottleneckResidualBlock('Discriminator.Down2', dim*1, dim*2, 3, output, resample='down')
        for i in xrange(6):
            output = BottleneckResidualBlock('Discriminator.16x16_{}'.format(i), dim*2, dim*2, 3, output, resample=None)
        output = BottleneckResidualBlock('Discriminator.Down3', dim*2, dim*4, 3, output, resample='down')
        for i in xrange(6):
            output = BottleneckResidualBlock('Discriminator.8x8_{}'.format(i), dim*4, dim*4, 3, output, resample=None)
        output = BottleneckResidualBlock('Discriminator.Down4', dim*4, dim*8, 3, output, resample='down')
        for i in xrange(6):
            output = BottleneckResidualBlock('Discriminator.4x4_{}'.format(i), dim*8, dim*8, 3, output, resample=None)

        output = tf.reshape(output, [-1, 4*4*8*dim])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

        return tf.reshape(output / 5., [-1])


    def FCDiscriminator(inputs, FC_DIM=512, n_layers=3):
        output = LeakyReLULayer('Discriminator.Input', OUTPUT_DIM, FC_DIM, inputs)
        for i in xrange(n_layers):
            output = LeakyReLULayer('Discriminator.{}'.format(i), FC_DIM, FC_DIM, output)
        output = lib.ops.linear.Linear('Discriminator.Out', FC_DIM, 1, output)

        return tf.reshape(output, [-1])

    def DCGANDiscriminator(inputs, dim=DIM, bn=True, nonlinearity=LeakyReLU):
        output = tf.reshape(inputs, [-1, 3, 64, 64])

        lib.ops.conv2d.set_weights_stdev(0.02)
        lib.ops.deconv2d.set_weights_stdev(0.02)
        lib.ops.linear.set_weights_stdev(0.02)

        output = lib.ops.conv2d.Conv2D('Discriminator.1', 3, dim, 5, output, stride=2)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.2', dim, 2*dim, 5, output, stride=2)
        if bn:
            output = Normalize('Discriminator.BN2', [0,2,3], output)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*dim, 4*dim, 5, output, stride=2)
        if bn:
            output = Normalize('Discriminator.BN3', [0,2,3], output)
        output = nonlinearity(output)

        output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*dim, 8*dim, 5, output, stride=2)
        if bn:
            output = Normalize('Discriminator.BN4', [0,2,3], output)
        output = nonlinearity(output)

        output = tf.reshape(output, [-1, 4*4*8*dim])
        output = lib.ops.linear.Linear('Discriminator.Output', 4*4*8*dim, 1, output)

        lib.ops.conv2d.unset_weights_stdev()
        lib.ops.deconv2d.unset_weights_stdev()
        lib.ops.linear.unset_weights_stdev()

        return tf.reshape(output, [-1])

    Generator, Discriminator = GeneratorAndDiscriminator()

    DEVICES = ['/gpu:{}'.format(i) for i in xrange(N_GPUS)]
    

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:

        all_real_data_conv = tf.placeholder(tf.int32, shape=[BATCH_SIZE, 3, 64, 64])
        if tf.__version__.startswith('1.'):
            split_real_data_conv = tf.split(all_real_data_conv, len(DEVICES))
        else:
            split_real_data_conv = tf.split(0, len(DEVICES), all_real_data_conv)
        gen_costs, disc_costs = [],[]
        
        # gamma placeholder for annealing
        if MODE == 'regularized_gan':
            gamma_plh = tf.placeholder(tf.float32, shape=(), name='gamma')

        for device_index, (device, real_data_conv) in enumerate(zip(DEVICES, split_real_data_conv)):
            with tf.device(device):

                real_data = tf.reshape(2*((tf.cast(real_data_conv, tf.float32)/255.)-.5), [BATCH_SIZE//len(DEVICES), OUTPUT_DIM])
                fake_data = Generator(BATCH_SIZE//len(DEVICES))

                disc_real = Discriminator(real_data)
                disc_fake = Discriminator(fake_data)

                if MODE == 'wgan':
                    gen_cost = -tf.reduce_mean(disc_fake)
                    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                elif MODE == 'wgan-gp':
                    gen_cost = -tf.reduce_mean(disc_fake)
                    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                    alpha = tf.random_uniform(
                        shape=[BATCH_SIZE//len(DEVICES),1],
                        minval=0.,
                        maxval=1.
                    )
                    differences = fake_data - real_data
                    interpolates = real_data + (alpha*differences)
                    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
                    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
                    disc_cost += LAMBDA*gradient_penalty
                
                elif MODE == 'gan':
                    try: # tf 1.0 (top) vs. pre-1.0 (bottom)
                        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))
                        disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
                        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
                    except Exception as e:
                        gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.ones_like(disc_fake)))
                        disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_fake, tf.zeros_like(disc_fake)))
                        disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(disc_real, tf.ones_like(disc_real)))
                    disc_cost /= 2.

                elif MODE == 'regularized_gan':
                    gen_cost  =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.ones_like(disc_fake)))
                    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real, labels=tf.ones_like(disc_real)))
                    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake, labels=tf.zeros_like(disc_fake)))
                    if not FLAGS.unreg:
                        disc_reg = Discriminator_Regularizer(disc_real, real_data, disc_fake, fake_data)
                        assert disc_cost.shape == disc_reg.shape
                        disc_cost += (gamma_plh/2.0)*disc_reg
                
                elif MODE == 'lsgan':
                    gen_cost = tf.reduce_mean((disc_fake - 1)**2)
                    disc_cost = (tf.reduce_mean((disc_real - 1)**2) + tf.reduce_mean((disc_fake - 0)**2))/2.

                else:
                    raise Exception()

                gen_costs.append(gen_cost)
                disc_costs.append(disc_cost)

        gen_cost = tf.add_n(gen_costs) / len(DEVICES)
        disc_cost = tf.add_n(disc_costs) / len(DEVICES)

        if MODE == 'wgan':
            gen_train_op  = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(gen_cost,
                                                      var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(learning_rate=5e-5).minimize(disc_cost,
                                                      var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

            clip_ops = []
            for var in lib.params_with_name('Discriminator'):
                clip_bounds = [-.01, .01]
                clip_ops.append(tf.assign(var, tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])))
            clip_disc_weights = tf.group(*clip_ops)

        elif MODE == 'wgan-gp':
            gen_train_op  = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(gen_cost,
                                                   var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0., beta2=0.9).minimize(disc_cost,
                                                   var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

        elif (MODE == 'regularized_gan') or (MODE == 'gan'):
            gen_train_op  = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(gen_cost,
                                                   var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
            disc_train_op = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(disc_cost,
                                                   var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

        elif MODE == 'lsgan':
            gen_train_op  = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(gen_cost,
                                                      var_list=lib.params_with_name('Generator'), colocate_gradients_with_ops=True)
            disc_train_op = tf.train.RMSPropOptimizer(learning_rate=1e-4).minimize(disc_cost,
                                                      var_list=lib.params_with_name('Discriminator.'), colocate_gradients_with_ops=True)

        else:
            raise Exception()

        # For generating samples
        fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, 128)).astype('float32'))
        all_fixed_noise_samples = []
        for device_index, device in enumerate(DEVICES):
            n_samples = BATCH_SIZE // len(DEVICES)
            all_fixed_noise_samples.append(Generator(n_samples, noise=fixed_noise[device_index*n_samples:(device_index+1)*n_samples]))
        if tf.__version__.startswith('1.'):
            all_fixed_noise_samples = tf.concat(all_fixed_noise_samples, axis=0)
        else:
            all_fixed_noise_samples = tf.concat(0, all_fixed_noise_samples)
        def generate_image(iteration):
            samples = session.run(all_fixed_noise_samples)
            samples = ((samples+1.)*(255.99/2)).astype('int32')
            lib.save_images.save_images(samples.reshape((BATCH_SIZE, 3, 64, 64)), log_dir + '/samples/samples_{}.png'.format(iteration))


        # Dataset iterator
        if FLAGS.dataset == 'ImageNet':
            train_gen, dev_gen = lib.small_imagenet.load(BATCH_SIZE, data_dir=DATA_DIR)
        elif FLAGS.dataset == 'lsun':
            train_gen, dev_gen = lib.load_lsun.load(BATCH_SIZE, data_dir=DATA_DIR, crop=True)
        elif FLAGS.dataset == 'celebA':
            train_gen, dev_gen = lib.load_celebA.load(BATCH_SIZE, data_dir=DATA_DIR, crop=True)

        def inf_train_gen():
            while True:
                for (images,) in train_gen():
                    yield images
        gen = inf_train_gen()

        # Save a batch of ground-truth samples
        _x = next(gen) ## gen().next()
        _x_r = session.run(real_data, feed_dict={real_data_conv: _x[:BATCH_SIZE//N_GPUS]})
        _x_r = ((_x_r+1.)*(255.99/2)).astype('int32')
        lib.save_images.save_images(_x_r.reshape((BATCH_SIZE//N_GPUS, 3, 64, 64)), log_dir + '/samples/samples_groundtruth.png')

        saver = tf.train.Saver(max_to_keep=1)

        # Train loop
        try:
            session.run(tf.global_variables_initializer())
        except:
            session.run(tf.initialize_all_variables())
        
        for iteration in xrange(ITERS):

            start_time = time.time()
            
            # ANNEALING (EXPONENTIAL DECAY from gamma to gamma*decay_factor)
            if MODE == 'regularized_gan':
                if FLAGS.annealing:
                    gamma = FLAGS.gamma*FLAGS.decay_factor**(iteration/(ITERS-1))
                else:
                    gamma = FLAGS.gamma

            # Train generator
            if iteration > 0:
                _ = session.run(gen_train_op)

            # Train critic
            if (MODE == 'regularized_gan') or (MODE == 'gan') or (MODE == 'lsgan'):
                disc_iters = FLAGS.disc_update_steps # 1
            else:
                disc_iters = CRITIC_ITERS

            for i in xrange(disc_iters):
                _data = next(gen) ## gen.next()
                if MODE == 'regularized_gan':
                    _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data, gamma_plh: gamma})
                else:
                    _disc_cost, _ = session.run([disc_cost, disc_train_op], feed_dict={all_real_data_conv: _data})

                if MODE == 'wgan':
                    _ = session.run([clip_disc_weights])

            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('time', time.time() - start_time)
            #print('gamma: {}'.format(gamma))

            if iteration % 200 == 199:
                t = time.time()
                dev_disc_costs = []
                for (images,) in dev_gen():
                    if MODE == 'regularized_gan':
                        _dev_disc_cost = session.run(disc_cost, feed_dict={all_real_data_conv: images, gamma_plh: gamma})
                    else:
                        _dev_disc_cost = session.run(disc_cost, feed_dict={all_real_data_conv: images})
                    dev_disc_costs.append(_dev_disc_cost)
                lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

                generate_image(iteration)
                saver.save(session, os.path.join(checkpoint_dir, "ckpt"), global_step=iteration)

            if (iteration < 5) or (iteration % 200 == 199):
                lib.plot.flush(log_dir)

            lib.plot.tick()


if __name__ == '__main__':
    main()
