from __future__ import division
import numpy as np
import tensorflow as tf
from six.moves import xrange
from glob import glob
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import math
import time
import os

from ops import *
from utils import *


class DCGAN(object):
# -----------------------------------------------------------------------------------
#     JS-Regularizer
# -----------------------------------------------------------------------------------
  def Discriminator_Regularizer(self, D1, D1_logits, D1_arg, D2, D2_logits, D2_arg):
    with tf.name_scope('disc_reg'):
      grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
      grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
      grad_D1_logits_norm = tf.norm( tf.reshape(grad_D1_logits, [self.batch_size,-1]) , axis=1, keep_dims=True)
      grad_D2_logits_norm = tf.norm( tf.reshape(grad_D2_logits, [self.batch_size,-1]) , axis=1, keep_dims=True)
    
      #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
      print('grad_D1_logits_norm.shape {} != D1.shape {}'.format(grad_D1_logits_norm.shape, D1.shape))
      print('grad_D2_logits_norm.shape {} != D2.shape {}'.format(grad_D2_logits_norm.shape, D2.shape))
      assert grad_D1_logits_norm.shape == D1.shape
      assert grad_D2_logits_norm.shape == D2.shape
    
      reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
      reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
    
      self.disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
      
      # various summaries
      self.reduce_mean_grad_D1_logits_norm = tf.reduce_mean(grad_D1_logits_norm)
      self.reduce_mean_grad_D2_logits_norm = tf.reduce_mean(grad_D2_logits_norm)
      self.reduce_mean_grad_D_logits_norm = self.reduce_mean_grad_D1_logits_norm + self.reduce_mean_grad_D2_logits_norm
      self.reduce_mean_reg_D1 = tf.reduce_mean(reg_D1)
      self.reduce_mean_reg_D2 = tf.reduce_mean(reg_D2)
    
      tf.summary.scalar("grad_D1_logits_norm", self.reduce_mean_grad_D1_logits_norm, family="disc")
      tf.summary.scalar("grad_D2_logits_norm", self.reduce_mean_grad_D2_logits_norm, family="disc")
      tf.summary.scalar("grad_D_logits_norm", self.reduce_mean_grad_D_logits_norm, family="disc")
      tf.summary.scalar("D1_regularizer", self.reduce_mean_reg_D1, family="disc")
      tf.summary.scalar("D2_regularizer", self.reduce_mean_reg_D2, family="disc")
      tf.summary.scalar("disc_regularizer", self.disc_regularizer, family="disc")
    
      grad_D1_norm = tf.norm( tf.reshape(tf.gradients(D1, D1_arg)[0], [self.batch_size,-1]) , axis=1, keep_dims=True)
      grad_D2_norm = tf.norm( tf.reshape(tf.gradients(D2, D2_arg)[0], [self.batch_size,-1]) , axis=1, keep_dims=True)
      self.reduce_mean_grad_D1_norm = tf.reduce_mean(grad_D1_norm)
      self.reduce_mean_grad_D2_norm = tf.reduce_mean(grad_D2_norm)
      self.reduce_mean_grad_D_norm = self.reduce_mean_grad_D1_norm + self.reduce_mean_grad_D2_norm
      
      tf.summary.scalar("grad_D1_norm", self.reduce_mean_grad_D1_norm, family="disc")
      tf.summary.scalar("grad_D2_norm", self.reduce_mean_grad_D2_norm, family="disc")
      tf.summary.scalar("grad_D_norm", self.reduce_mean_grad_D_norm, family="disc")
    
      return self.disc_regularizer


# -----------------------------------------------------------------------------------
#     Initialization
# -----------------------------------------------------------------------------------
  def __init__(self, sess, log_dir, flags):
      
    self.sess = sess
    self.dataset = flags.dataset
    self.datadir = "./data"
    
    self.batch_norm = True
    
    self.batch_size = flags.batch_size
    self.sample_num = flags.batch_size
    
    self.log_dir=log_dir
    
    self.input_fname_pattern = flags.input_fname_pattern
    self.checkpoint_dir = flags.checkpoint_dir
    
    self.z_dim = 100
    if self.dataset == 'mnist':
      self.y_dim=10
    else:
      self.y_dim=None
    
    self.gf_dim = 64
    self.df_dim = 64

    self.gfc_dim = 1024
    self.dfc_dim = 1024
    
    if self.dataset == 'mnist':
      self.input_height = 28
      self.input_width = 28
    elif self.dataset == 'cifar10':
      self.input_height = 32
      self.input_width = 32
    elif self.dataset == 'celebA':
      self.input_height = 108
      self.input_width = 108
    else:
      self.input_height = flags.input_height
      self.input_width = flags.input_width
    self.output_height = flags.output_height
    self.output_width = flags.output_width

    self.crop = True

    if self.dataset == 'mnist':
      self.c_dim = 1
    elif self.dataset == 'cifar10' or 'celebA':
      self.c_dim = 3
    else:
      data = glob(os.path.join(self.datadir, self.dataset, self.input_fname_pattern))
      if len(imread(data[0]).shape) >= 3:
        self.c_dim = imread(data[0]).shape[-1]
        print('c_dim = {}'.format(self.c_dim))
      else:
        self.c_dim = 1
      
    self.grayscale = (self.c_dim == 1)
    
    if self.crop:
      self.image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      self.image_dims = [self.input_height, self.input_width, self.c_dim]

    self.build_graph(flags)


# -----------------------------------------------------------------------------------
#     Computational Graph (Placeholders & GAN Objective)
# -----------------------------------------------------------------------------------
  def build_graph(self, flags):
      
    self.inputs = tf.placeholder(tf.float32, [self.batch_size] + self.image_dims, name='images')

    if self.y_dim:
      self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')
    else:
      self.y = None

    self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim], name='z')


    # gamma placeholder for annealing
    self.gamma = tf.placeholder(tf.float32, shape=(), name='gamma')
    tf.summary.scalar("gamma", self.gamma)

    # bn_train: True during training, False during inference!
    self.bn_train = tf.placeholder(tf.bool, shape=(), name='bn_train')
    

    self.G                  = self.generator(self.z, self.y, sum=True)
    self.D1, self.D1_logits = self.discriminator(self.inputs, self.y, sum=True)
    self.D2, self.D2_logits = self.discriminator(self.G, self.y, reuse=True, sum=True)

    #tf.summary.image("G", self.G, family="gen")
    tf.summary.histogram("D1", self.D1, family="disc")
    tf.summary.histogram("D2", self.D2, family="disc")
    

    self.d_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D1_logits, labels=tf.ones_like(self.D1))
                                 +tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.zeros_like(self.D2)))

    self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D2_logits, labels=tf.ones_like(self.D2)))

    tf.summary.scalar("unreg_disc_loss", self.d_loss, family="disc")
    tf.summary.scalar("unreg_gen_loss", self.g_loss, family="gen")

    if not flags.unreg:
      self.d_reg = self.Discriminator_Regularizer(self.D1, self.D1_logits, self.inputs, self.D2, self.D2_logits, self.G)
      assert self.d_loss.shape == self.d_reg.shape
      self.d_loss += (self.gamma/2.0)*self.d_reg

      tf.summary.scalar("disc_loss", self.d_loss, family="disc")


    self.g_iterations = tf.Variable(0, name="g_iterations", trainable=False)
    self.d_iterations = tf.Variable(0, name="d_iterations", trainable=False)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver(max_to_keep=1)


    self.d_train_op = self.optimizer(self.d_loss, self.d_vars, flags.disc_learning_rate, self.d_iterations, flags)
    self.g_train_op = self.optimizer(self.g_loss, self.g_vars, flags.gen_learning_rate, self.g_iterations, flags)

    if self.checkpoint_dir is "None":
      self.sess.run(tf.global_variables_initializer())
              
    self.summary_op = tf.summary.merge_all()
    self.summary_writer = tf.summary.FileWriter(self.log_dir+'/summaries', self.sess.graph)
      
      

# -----------------------------------------------------------------------------------
#     Optimizer
# -----------------------------------------------------------------------------------
# If update ops are placed in tf.GraphKeys.UPDATE_OPS for
# tf.contrib.layers.batch_norm( ... updates_collections=tf.GraphKeys.UPDATE_OPS ...)
# they need to be added as a dependency to the optimizer_op as follows:
#   update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
#   with tf.control_dependencies(update_ops):
# We use updates_collections=None and force updates in place instead:
# -----------------------------------------------------------------------------------
  def optimizer(self, loss, var_list, learning_rate, global_step, flags):
    
    if flags.rmsprop:
      opt = tf.train.RMSPropOptimizer(learning_rate)
    else:
      opt = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    optimizer_op = opt.minimize(loss, global_step=global_step, var_list=var_list)
    
    return optimizer_op


# -----------------------------------------------------------------------------------
#     Training
# -----------------------------------------------------------------------------------
  def train(self, flags):

    if not os.path.exists(self.log_dir+'/samples'):
      os.makedirs(self.log_dir+'/samples')


    if not self.checkpoint_dir == "None":
      load_bool, resume_step = self.load_ckpt()
      if load_bool:
        print("Resume from Step: {0:5d}".format(resume_step))
        self.sess.run(self.g_iterations.assign(resume_step))
        print(" [*] Load SUCCESS")
      else:
        print(" [!] Load failed...")


    if self.dataset == 'mnist':
      data, labels = self.load_mnist()
      fixed_images, fixed_labels = data[0:self.sample_num], labels[0:self.sample_num]
      data, labels = data[self.sample_num:], labels[self.sample_num:]
    elif self.dataset == 'cifar10':
      data = self.load_cifar10()
      fixed_images = data[0:self.sample_num]
      data = data[self.sample_num:]
    else:
      pathnames = glob(os.path.join(self.datadir, self.dataset, self.input_fname_pattern))
      data = [
        get_image(image,
                  input_height=self.input_height,
                  input_width=self.input_width,
                  resize_height=self.output_height,
                  resize_width=self.output_width,
                  crop=self.crop,
                  grayscale=self.grayscale) for image in pathnames]
      if (self.grayscale):
          data = np.array(data).astype(np.float32)[:, :, :, None]
      else:
          data = np.array(data).astype(np.float32)
      np.random.shuffle(data)
      fixed_images = data[0:self.sample_num]
      data = data[self.sample_num:]


    if flags.gaussian_prior:
      fixed_z = np.random.normal(0, 1, size=[self.sample_num , self.z_dim]).astype(np.float32)
    else:
      fixed_z = np.random.uniform(-1, 1, size=[self.sample_num , self.z_dim]).astype(np.float32)


    # LOOP
    bs = self.batch_size
    d_ups = flags.disc_update_steps
    batch_idxs = min(len(data), flags.dataset_size) // (bs*d_ups)

    for epoch in xrange(flags.epochs):
        
      # re-shuffle
      if self.dataset == 'mnist':
        np.random.seed(epoch)
        np.random.shuffle(data)
        np.random.seed(epoch)
        np.random.shuffle(labels)
      else:
        np.random.shuffle(data)


      for idx in xrange(batch_idxs):
        start_time = time.time()
        
        # ANNEALING (EXPONENTIAL DECAY from gamma to decay_factor*gamma)
        if flags.annealing:
          gamma = flags.gamma*flags.decay_factor**((epoch*batch_idxs+idx)/(flags.epochs*batch_idxs-1))
        else:
          gamma = flags.gamma
    
    
        # Update D network
        batch_images = data[idx*bs*d_ups:(idx+1)*bs*d_ups]
        if self.dataset == 'mnist':
          batch_labels = labels[idx*bs*d_ups:(idx+1)*bs*d_ups]

        if flags.gaussian_prior:
          batch_z       = np.random.normal(0, 1, size=[bs*d_ups, self.z_dim]).astype(np.float32)
        else:
          batch_z       = np.random.uniform(-1, 1, size=[bs*d_ups, self.z_dim]).astype(np.float32)

        for dstep in xrange(flags.disc_update_steps):
            
          d_feed_dict = {self.inputs:     batch_images[dstep*bs:(dstep+1)*bs],
                         self.z:          batch_z[dstep*bs:(dstep+1)*bs],
                         self.gamma:      gamma,
                         self.bn_train:   True}
          if self.dataset == 'mnist':
            d_feed_dict[self.y] = batch_labels[dstep*bs:(dstep+1)*bs]

          self.sess.run(self.d_train_op, feed_dict=d_feed_dict)


        # Update G network
        if self.dataset == 'mnist':
          # generate some random labels
          batch_labels = np.zeros((bs, 10))
          batch_labels[np.arange(bs), np.random.choice(10, bs)] = 1

        if flags.gaussian_prior:
          batch_z       = np.random.normal(0, 1, size=[bs, self.z_dim]).astype(np.float32)
        else:
          batch_z       = np.random.uniform(-1, 1, size=[bs, self.z_dim]).astype(np.float32)

        g_feed_dict = {self.z:          batch_z,
                       self.gamma:      gamma,
                       self.bn_train:   True}
        if self.dataset == 'mnist':
          g_feed_dict[self.y] = batch_labels

        self.sess.run(self.g_train_op, feed_dict=g_feed_dict)


        # Summarize state of the networks and save generated images
        current_step = tf.train.global_step(self.sess, self.g_iterations)

        if ((np.mod(current_step, 100) == 0) or (epoch == 0 and idx == 0)):

          sum_dict = {self.inputs:     fixed_images,
                      self.z:          fixed_z,
                      self.gamma:      gamma,
                      self.bn_train:   False}
          if self.dataset == 'mnist':
            sum_dict[self.y] = fixed_labels

          samples, summary_str = self.sess.run([self.G, self.summary_op], feed_dict=sum_dict)
          self.summary_writer.add_summary(summary_str, current_step)
          self.summary_writer.flush()

          try:
            grid_h = int(np.ceil(np.sqrt(samples.shape[0])))
            grid_w = int(np.floor(np.sqrt(samples.shape[0])))
            save_images(samples, [grid_h, grid_w], self.dataset, self.log_dir+'/samples/train_epoch{:02d}_run{:04d}_gups{:05d}.png'.format(epoch+1, idx+1, current_step))
          except:
            print("ERROR: Couldn't Save Images!")

          if current_step > 0:
            self.save(current_step)


        print("Epoch: {0:2d}/{1:2d} [Round: {2:4d}/{3:4d}] gamma: {4:.4f}, time: {5:.2f}".format(epoch, flags.epochs, idx, batch_idxs, gamma, time.time() - start_time))

    # Save latest state
    self.save(current_step)
    self.generate(flags, option=0)

# -----------------------------------------------------------------------------------
#     Networks
# -----------------------------------------------------------------------------------
  # image.shape = [batch_size, self.output_height, self.output_width, self.c_dim]
  def discriminator(self, image, y=None, reuse=False, sum=False, fam='disc'):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()
      
      # Batch Normalization Layers vs. Identity
      if self.batch_norm:
        self.d_l1   = batch_norm(name='d_bn1')
        self.d_l2   = batch_norm(name='d_bn2')
        if not self.y_dim:
          self.d_l3 = batch_norm(name='d_bn3')
      else:
        self.d_l1   = identity_op(name='d_id1')
        self.d_l2   = identity_op(name='d_id2')
        if not self.y_dim:
          self.d_l3 = identity_op(name='d_id3')

      # Unconditional GAN
      if not self.y_dim:
        # conv2d(input, output_dim, name)
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv', sum=sum))
        h1 = lrelu(self.d_l1(conv2d(h0, self.df_dim*2, name='d_h1_conv', sum=sum), train=self.bn_train))
        h2 = lrelu(self.d_l2(conv2d(h1, self.df_dim*4, name='d_h2_conv', sum=sum), train=self.bn_train))
        h3 = lrelu(self.d_l3(conv2d(h2, self.df_dim*8, name='d_h3_conv', sum=sum), train=self.bn_train))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, name='d_h4_lin', sum=sum)
        
        if sum:
          tf.summary.histogram('d_h0_conv', h0, family=fam)
          tf.summary.histogram('d_h1_conv', h1, family=fam)
          tf.summary.histogram('d_h2_conv', h2, family=fam)
          tf.summary.histogram('d_h3_conv', h3, family=fam)
          tf.summary.histogram('d_h4_sigm', tf.nn.sigmoid(h4), family=fam)

        return tf.nn.sigmoid(h4), h4
      # Conditional GAN
      else:
        # concatenate color channels with one-hot labels
        y_rs = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        x = conv_cond_concat(image, y_rs)
        # x.shape == [batch_size, self.output_height, self.output_width, self.c_dim + self.y_dim]

        h0  = lrelu(conv2d(x, self.c_dim + self.y_dim, name='d_h0_conv', sum=sum))
        h0_ = conv_cond_concat(h0, y_rs)

        h1  = lrelu(self.d_l1(conv2d(h0_, self.df_dim + self.y_dim, name='d_h1_conv', sum=sum), train=self.bn_train))
        h1  = tf.reshape(h1, [self.batch_size, -1])
        h1_ = concat([h1, y], 1)
        
        h2  = lrelu(self.d_l2(linear(h1_, self.dfc_dim, name='d_h2_lin', sum=sum), train=self.bn_train))
        h2_ = concat([h2, y], 1)

        h3  = linear(h2_, 1, name='d_h3_lin', sum=sum)
        
        if sum:
          tf.summary.histogram('d_h0_conv', h0, family=fam)
          tf.summary.histogram('d_h1_conv', h1, family=fam)
          tf.summary.histogram('d_h2_lin', h2, family=fam)
          tf.summary.histogram('d_h3_sigm', tf.nn.sigmoid(h3), family=fam)
        
        return tf.nn.sigmoid(h3), h3


  def generator(self, z, y=None, reuse=False, sum=False, fam='gen'):
    with tf.variable_scope("generator") as scope:
      if reuse:
        scope.reuse_variables()
      
      # Batch Normalization Layers vs. Identity
      if self.batch_norm:
        self.g_l0   = batch_norm(name='g_bn0')
        self.g_l1   = batch_norm(name='g_bn1')
        self.g_l2   = batch_norm(name='g_bn2')
        if not self.y_dim:
          self.g_l3 = batch_norm(name='g_bn3')
      else:
        self.g_l0   = identity_op(name='g_id0')
        self.g_l1   = identity_op(name='g_id1')
        self.g_l2   = identity_op(name='g_id2')
        if not self.y_dim:
          self.g_l3 = identity_op(name='g_id3')
    
      # Unconditional GAN
      if not self.y_dim:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        # project `z` and reshape
        h0 = tf.reshape(linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', sum=sum),[-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_l0(h0, train=self.bn_train))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', sum=sum)
        h1 = tf.nn.relu(self.g_l1(h1, train=self.bn_train))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', sum=sum)
        h2 = tf.nn.relu(self.g_l2(h2, train=self.bn_train))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', sum=sum)
        h3 = tf.nn.relu(self.g_l3(h3, train=self.bn_train))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', sum=sum)
        h4 = tf.nn.tanh(h4)
        
        if sum:
          tf.summary.histogram('g_h0_lin', h0, family=fam)
          tf.summary.histogram('g_h1', h1, family=fam)
          tf.summary.histogram('g_h2', h2, family=fam)
          tf.summary.histogram('g_h3', h3, family=fam)
          tf.summary.histogram('g_h4_tanh', h4, family=fam)

        return h4
      # Conditional GAN
      else:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_h4 = int(s_h/2), int(s_h/4)
        s_w2, s_w4 = int(s_w/2), int(s_w/4)

        y_rs = tf.reshape(y, [self.batch_size, 1, 1, self.y_dim])
        z = concat([z, y], 1)
        # z.shape == [batch_size, z_dim + y_dim]

        h0  = tf.nn.relu(self.g_l0(linear(z, self.gfc_dim, name='g_h0_lin', sum=sum), train=self.bn_train))
        h0_ = concat([h0, y], 1)

        h1  = tf.nn.relu(self.g_l1(linear(h0_, self.gf_dim*2*s_h4*s_w4, name='g_h1_lin', sum=sum), train=self.bn_train))
        h1  = tf.reshape(h1, [self.batch_size, s_h4, s_w4, self.gf_dim * 2])
        h1_ = conv_cond_concat(h1, y_rs)

        h2  = tf.nn.relu(self.g_l2(deconv2d(h1_, [self.batch_size, s_h2, s_w2, self.gf_dim * 2], name='g_h2', sum=sum), train=self.bn_train))
        h2_ = conv_cond_concat(h2, y_rs)
        
        h3 = deconv2d(h2_, [self.batch_size, s_h, s_w, self.c_dim], name='g_h3', sum=sum)
        h3 = tf.nn.sigmoid(h3)
        
        if sum:
          tf.summary.histogram('g_h0_lin', h0, family=fam)
          tf.summary.histogram('g_h1_lin', h1, family=fam)
          tf.summary.histogram('g_h2', h2, family=fam)
          tf.summary.histogram('g_h3_sigm', h3, family=fam)

        return h3



# -----------------------------------------------------------------------------------
#     Load & Save
# -----------------------------------------------------------------------------------
  def load_mnist(self):
    data_dir = os.path.join(self.datadir, self.dataset)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    np.random.seed(784)
    np.random.shuffle(X)
    np.random.seed(784)
    np.random.shuffle(y)
    
    y_hot = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_hot[i,y[i]] = 1.0
    
    return X/255.,y_hot
    # X in [0,1]^784 (sigmoid generator activation)
    
    
  def load_cifar10(self):
    data_dir = os.path.join(self.datadir, self.dataset)
  
    data_batch_1_path = os.path.join(data_dir, "data_batch_1")
    data_batch_1 = unpickle(data_batch_1_path)[b'data'].astype(np.float)
  
    data_batch_2_path = os.path.join(data_dir, "data_batch_2")
    data_batch_2 = unpickle(data_batch_2_path)[b'data'].astype(np.float)
  
    data_batch_3_path = os.path.join(data_dir, "data_batch_3")
    data_batch_3 = unpickle(data_batch_3_path)[b'data'].astype(np.float)

    data_batch_4_path = os.path.join(data_dir, "data_batch_4")
    data_batch_4 = unpickle(data_batch_4_path)[b'data'].astype(np.float)
        
    data_batch_5_path = os.path.join(data_dir, "data_batch_5")
    data_batch_5 = unpickle(data_batch_5_path)[b'data'].astype(np.float)
  
    # db.shape == [50000,32*32*3]
    # Each row of db stores a 32x32 3-colour image.
    # The first 32*32=1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue.
    # The 32*32 pixels stored in row-major order [1st row, 2nd row, ..., 32nd row]
    db = np.concatenate((data_batch_1,data_batch_2,data_batch_3,data_batch_4,data_batch_5), axis=0)
    db = db.reshape([-1,3,32*32]).swapaxes(1,2).reshape([-1,32,32,3])
    
    np.random.shuffle(db)
  
    return db/127.5 - 1.
    # X in [-1,1]^3072 (tanh generator activation)


    
  def save(self, step):
    checkpoint_dir = os.path.join(self.log_dir, "checkpoints")

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess, os.path.join(checkpoint_dir, "ckpt"), global_step=step)
      

  def load_ckpt(self):
    import re
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(self.checkpoint_dir, ckpt_name))
      resume_step = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Successfully read {}".format(ckpt_name))
      return True, resume_step
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0




# -----------------------------------------------------------------------------------
#     Generate Samples
# -----------------------------------------------------------------------------------
  def generate(self, flags, option=0):

    image_frame_dim = int(math.ceil(self.batch_size**.5))
    if option == 0:
      for idx in xrange(10):
        if flags.gaussian_prior:
          z = np.random.normal(0, 1, size=[self.batch_size , self.z_dim]).astype(np.float32)
        else:
          z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)
        if self.dataset == "mnist":
          y = np.random.choice(10, self.batch_size)
          y_hot = np.zeros((self.batch_size, 10))
          y_hot[np.arange(self.batch_size), y] = 1

          samples = self.sess.run(self.G, feed_dict={self.z: z, self.y: y_hot, self.bn_train: False})
        else:
          samples = self.sess.run(self.G, feed_dict={self.z: z, self.bn_train: False})
        save_images(samples, [image_frame_dim, image_frame_dim], self.dataset, self.log_dir+'/samples/test_%s.png' % (idx))


    elif option == 1:
      if not os.path.exists(self.log_dir+'/samps_10k'):
        os.makedirs(self.log_dir+'/samps_10k')
      labels = []
      for idx in xrange(int(math.ceil(10000/self.batch_size))):
        if flags.gaussian_prior:
          z = np.random.normal(0, 1, size=[self.batch_size , self.z_dim]).astype(np.float32)
        else:
          z = np.random.uniform(-1.0, 1.0, size=(self.batch_size, self.z_dim)).astype(np.float32)
        if self.dataset == "mnist":
          y = np.random.choice(10, self.batch_size)
          y_hot = np.zeros((self.batch_size, 10))
          y_hot[np.arange(self.batch_size), y] = 1
          labels.append(y_hot)
                
          samples = self.sess.run(self.G, feed_dict={self.z: z, self.y: y_hot, self.bn_train: False})
        else:
          samples = self.sess.run(self.G, feed_dict={self.z: z, self.bn_train: False})

        if self.dataset == "mnist":
          for i in xrange(self.batch_size):
            scipy.misc.imsave(self.log_dir+'/samps_10k/image_{}.png'.format(int(idx*self.batch_size+i)), inverse_transform(samples[i][:,:,0], self.dataset).astype(np.uint8))
        else:
          for i in xrange(self.batch_size):
            scipy.misc.imsave(self.log_dir+'/samps_10k/image_{}.png'.format(int(idx*self.batch_size+i)), inverse_transform(samples[i], self.dataset).astype(np.uint8))
    
      if self.dataset == "mnist":
        labels = np.concatenate(labels, axis=0)
        np.save(self.log_dir+'/samps_10k/labels.npy', labels)

