#
# Stabilizing Training of Generative Adversarial Networks through Regularization
# https://arxiv.org/pdf/1705.09367.pdf
#
# Inspired by https://github.com/poolio/unrolled_gan
#
import tensorflow as tf
import tensorflow.contrib.distributions as tfcds
import numpy as np
from scipy import stats, linalg, misc
from math import pi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse
import time
import sys
import os


Z_DIM = 2
GAMMA = 0.1
DECAY_FACTOR = 0.01
MOG_STDDEV=0.01
GEN_LEARNING_RATE = 1e-3
DISC_LEARNING_RATE = 1e-3
DISC_PRETRAIN_STEPS = 0
DISC_UPDATE_STEPS = 1
GEN_UPDATE_STEPS = 1
BATCH_SIZE = 512
VIZ_BATCHES = 10
MAX_STEPS = 100000
PLOT_FREQ = 10000
NUM_RUNS = 3



FLAGS = None
parser = argparse.ArgumentParser()
parser.add_argument('--z_dim', type=int, default=Z_DIM, help='Dimension of Latent Variable.')
parser.add_argument('--unreg', action='store_true', help='Turn Regularization Off.')
parser.add_argument('--alt_gen_loss', action='store_true', help='Use Alternative Generator Loss.')
parser.add_argument('--annealing', action='store_true', help='Annealing gamma*decay_factor^t/T [False].')
parser.add_argument('--decay_factor', type=float, default=DECAY_FACTOR, help='Exponential annealing decay factor.')
parser.add_argument('--sgd', action='store_true', help='SGD optimizer (Adam by default).')
parser.add_argument('--rmsprop', action='store_true', help='RMSProp optimizer (Adam by default).')
parser.add_argument('--wall_clock', action='store_true', help='Count every update of both G and D (by default generator iterations only).')
parser.add_argument('--uniform_prior', action='store_true', help='Uniform prior (Gaussian by default).')
parser.add_argument('--gamma', type=float, default=GAMMA, help='Gradient-Regularizer bandwidth.')
parser.add_argument('--disc_learning_rate', type=float, default=DISC_LEARNING_RATE, help='(Initial) Learning Rate.')
parser.add_argument('--gen_learning_rate', type=float, default=GEN_LEARNING_RATE, help='(Initial) Learning Rate.')
parser.add_argument('--disc_pretrain_steps', type=int, default=DISC_PRETRAIN_STEPS, help='Discriminator Pretrain Steps.')
parser.add_argument('--disc_update_steps', type=int, default=DISC_UPDATE_STEPS, help='Discriminator Update Steps.')
parser.add_argument('--gen_update_steps', type=int, default=GEN_UPDATE_STEPS, help='Generator Update Steps.')
parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='Batch size.')
parser.add_argument('--viz_batches', type=int, default=VIZ_BATCHES, help='Number of Batches for Visualization.')
parser.add_argument('--max_steps', type=int, default=MAX_STEPS, help='Max Steps.')
parser.add_argument('--plot_freq', type=int, default=PLOT_FREQ, help='Plot frequency.')
parser.add_argument('--num_runs', type=int, default=NUM_RUNS, help='Number of Runs.')
parser.add_argument('--root_dir', type=str, default='RUN_STATS', help='root directory [RUN_STATS]')
FLAGS, unparsed = parser.parse_known_args()




# Strings

file_name = time.strftime("%Y_%m_%d_%H%M", time.localtime())
if FLAGS.unreg:
    file_name += "_unreg_gan"
else:
    file_name += "_regularized_gan_"+str(FLAGS.gamma)+"gamma"
if FLAGS.annealing:
    file_name += "_annealing_"+str(FLAGS.decay_factor)+"decayfactor"
if FLAGS.alt_gen_loss:
    file_name += "_altgenloss"
if FLAGS.sgd:
    file_name += "_sgd"
elif FLAGS.rmsprop:
    file_name += "_rmsprop"
else:
    file_name += "_adam"
if FLAGS.disc_pretrain_steps > 0:
    file_name += "_"+str(FLAGS.disc_pretrain_steps)+"pretraindsteps"
file_name += "_"+str(FLAGS.disc_update_steps)+"dsteps"
file_name += "_"+str(FLAGS.disc_learning_rate)+"dlnr"
if FLAGS.gen_update_steps > 1:
    file_name += "_"+str(FLAGS.gen_update_steps)+"gsteps"
file_name += "_"+str(FLAGS.gen_learning_rate)+"glnr"
if FLAGS.uniform_prior:
    file_name += "_uniformprior"
if FLAGS.num_runs > 1:
    file_name += "_"+str(FLAGS.num_runs)+"runs"
log_dir = os.path.abspath(os.path.join(os.path.curdir, FLAGS.root_dir, file_name))



# -----------------------------------------------------------------------------------
#     Mixture of Gaussians
# -----------------------------------------------------------------------------------
def RodriguesRotationMatrix(axis, theta):
    with tf.name_scope('RodriguesRotationMatrix'):
        assert np.size(axis)== 3
        # Rotation generator
        K = tf.constant([[0., -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0.]])
        R = tf.eye(3) + tf.sin(theta)*K + (1-tf.cos(theta))*tf.matmul(K,K)
        return R

def RotateAndTranslate(data):
    with tf.name_scope('RotateAndTranslate'):
        axis = np.array([1.,-1.,0.])/np.sqrt(2.)
        theta = pi/4
        R = RodriguesRotationMatrix(axis, theta)
        t = tf.ones(3)/tf.sqrt(3.)
        transform = tf.add(tf.matmul(data, R), t)
        return transform

def MixtureOfGaussians(batch_size, components=7, stddev=MOG_STDDEV, radius=1.0, RAT=True):
    with tf.name_scope('MixtureOfGaussians'):
        thetas = np.linspace(0, 2*pi, components+1)[:-1]
        xs, ys = radius * np.sin(thetas), radius * np.cos(thetas)
        cat = tfcds.Categorical(np.zeros(components))
        comps = [tfcds.MultivariateNormalDiag([xi, yi], [stddev, stddev]) for xi, yi in zip(xs, ys)]
        mixture = tfcds.Mixture(cat, comps)
        # Embedding 2D Mixture in 3D space
        E = tf.constant([[1.0, 0.0, 0.0],[0.0, 1.0, 0.0]])
        embedding = tf.matmul(mixture.sample(batch_size),E)
        if RAT==True:
            submanifold = RotateAndTranslate(embedding)
            return submanifold
        else:
            return embedding



# -----------------------------------------------------------------------------------
#     Neural Nets
# -----------------------------------------------------------------------------------
class Generator(object):
    def __init__(self, z, hidden_units=128, scope="generator"):
        stddev = 0.1
        weight_initializer = tf.truncated_normal_initializer(stddev=stddev)
        bias_initializer = tf.constant_initializer(0.0)
        with tf.variable_scope(scope):
            self.W1 = tf.get_variable('W1', [FLAGS.z_dim, hidden_units], initializer=weight_initializer)
            self.b1 = tf.get_variable('b1', [hidden_units], initializer=bias_initializer)
            self.W2 = tf.get_variable('W2', [hidden_units, hidden_units], initializer=weight_initializer)
            self.b2 = tf.get_variable('b2', [hidden_units], initializer=bias_initializer)
            self.W3 = tf.get_variable('W3', [hidden_units, 3], initializer=weight_initializer)
            self.b3 = tf.get_variable('b3', [3], initializer=bias_initializer)
        self.h1 = tf.nn.tanh(tf.matmul(z, self.W1) + self.b1)
        self.h2 = tf.nn.tanh(tf.matmul(self.h1, self.W2) + self.b2)
        self.x = tf.matmul(self.h2, self.W3) + self.b3

class Discriminator(object):
    def __init__(self, x, hidden_units=128, reuse=False, scope="discriminator"):
        stddev = 0.1
        weight_initializer = tf.truncated_normal_initializer(stddev=stddev)
        bias_initializer = tf.constant_initializer(0.0)
        with tf.variable_scope(scope, reuse=reuse):
            self.W1 = tf.get_variable('W1', [3, hidden_units], initializer=weight_initializer)
            self.b1 = tf.get_variable('b1', [hidden_units], initializer=bias_initializer)
            self.W2 = tf.get_variable('W2', [hidden_units, hidden_units], initializer=weight_initializer)
            self.b2 = tf.get_variable('b2', [hidden_units], initializer=bias_initializer)
            self.W3 = tf.get_variable('W3', [hidden_units, 1], initializer=weight_initializer)
            self.b3 = tf.get_variable('b3', [1], initializer=bias_initializer)
        self.h1 = tf.nn.tanh(tf.matmul(x, self.W1) + self.b1)
        self.h2 = tf.nn.tanh(tf.matmul(self.h1, self.W2) + self.b2)
        self.p_logits = tf.matmul(self.h2, self.W3) + self.b3
        self.p = tf.nn.sigmoid(self.p_logits)


# -----------------------------------------------------------------------------------
#     JS-Regularizer
# -----------------------------------------------------------------------------------
def Discriminator_Regularizer(D1, D1_logits, D1_arg, D2, D2_logits, D2_arg):
    with tf.name_scope('disc_reg'):
        grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
        grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
        grad_D1_logits_norm = tf.norm(grad_D1_logits, axis=1, keep_dims=True)
        grad_D2_logits_norm = tf.norm(grad_D2_logits, axis=1, keep_dims=True)

        #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
        assert grad_D1_logits_norm.shape == D1.shape
        assert grad_D2_logits_norm.shape == D2.shape

        reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
        reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))

        disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)

        # various summaries
        reduce_mean_grad_D1_logits_norm = tf.reduce_mean(grad_D1_logits_norm)
        reduce_mean_grad_D2_logits_norm = tf.reduce_mean(grad_D2_logits_norm)
        reduce_mean_grad_D_logits_norm = reduce_mean_grad_D1_logits_norm + reduce_mean_grad_D2_logits_norm
        reduce_mean_reg_D1 = tf.reduce_mean(reg_D1)
        reduce_mean_reg_D2 = tf.reduce_mean(reg_D2)

        tf.summary.scalar("grad_D1_logits_norm", reduce_mean_grad_D1_logits_norm, family="disc")
        tf.summary.scalar("grad_D2_logits_norm", reduce_mean_grad_D2_logits_norm, family="disc")
        tf.summary.scalar("grad_D_logits_norm", reduce_mean_grad_D_logits_norm, family="disc")
        tf.summary.scalar("D1_regularizer", reduce_mean_reg_D1, family="disc")
        tf.summary.scalar("D2_regularizer", reduce_mean_reg_D2, family="disc")
        tf.summary.scalar("disc_regularizer", disc_regularizer, family="disc")

        grad_D1_norm = tf.norm(tf.gradients(D1, D1_arg)[0], axis=1, keep_dims=True)
        grad_D2_norm = tf.norm(tf.gradients(D2, D2_arg)[0], axis=1, keep_dims=True)
        reduce_mean_grad_D1_norm = tf.reduce_mean(grad_D1_norm)
        reduce_mean_grad_D2_norm = tf.reduce_mean(grad_D2_norm)
        reduce_mean_grad_D_norm = reduce_mean_grad_D1_norm + reduce_mean_grad_D2_norm

        tf.summary.scalar("grad_D1_norm", reduce_mean_grad_D1_norm, family="disc")
        tf.summary.scalar("grad_D2_norm", reduce_mean_grad_D2_norm, family="disc")
        tf.summary.scalar("grad_D_norm", reduce_mean_grad_D_norm, family="disc")

        return disc_regularizer


# -----------------------------------------------------------------------------------
#     Optimizer
# -----------------------------------------------------------------------------------
def train(loss, learning_rate, global_step, var_list):
    if FLAGS.sgd:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    elif FLAGS.rmsprop:
        optimizer = tf.train.RMSPropOptimizer(learning_rate)
    else:
        optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5)
    train_op = optimizer.minimize(loss, global_step=global_step, var_list=var_list)
    return train_op



# -----------------------------------------------------------------------------------
#     Computational Graph
# -----------------------------------------------------------------------------------

# MoG & Generator Samples

mog_x = MixtureOfGaussians(FLAGS.batch_size)

if FLAGS.uniform_prior:
    z = tfcds.Uniform(-tf.ones(FLAGS.z_dim), tf.ones(FLAGS.z_dim)).sample(FLAGS.batch_size)
else:
    z = tfcds.Normal(tf.zeros(FLAGS.z_dim), tf.ones(FLAGS.z_dim)).sample(FLAGS.batch_size)
gen_x = Generator(z).x


# Discriminator Scores

D1 = Discriminator(mog_x).p
D1_logits = Discriminator(mog_x, reuse=True).p_logits
D2 = Discriminator(gen_x, reuse=True).p
D2_logits = Discriminator(gen_x, reuse=True).p_logits

tf.summary.histogram("D1", D1, family="disc")
tf.summary.histogram("D2", D2, family="disc")


# gamma placeholder for annealing
gamma_plh = tf.placeholder(tf.float32, shape=(), name='gamma')
tf.summary.scalar("gamma", gamma_plh)


# GAN Objective

disc_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_logits, labels=tf.ones_like(D1))
                           +tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logits, labels=tf.zeros_like(D2)))

if FLAGS.alt_gen_loss:
    gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logits, labels=tf.ones_like(D2)))
else:
    gen_loss = tf.reduce_mean(-tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logits, labels=tf.zeros_like(D2)))

tf.summary.scalar("unreg_disc_loss", disc_loss, family="disc")
tf.summary.scalar("unreg_gen_loss", gen_loss, family="gen")

if not FLAGS.unreg:
    disc_reg = Discriminator_Regularizer(D1, D1_logits, mog_x, D2, D2_logits, gen_x)
    assert disc_loss.shape == disc_reg.shape
    disc_loss += (gamma_plh/2.0)*disc_reg

    tf.summary.scalar("disc_loss", disc_loss, family="disc")


# Train-ops

wall_clock = tf.Variable(0, name="wall_clock", trainable=False)
gen_iterations = tf.Variable(0, name="gen_iterations", trainable=False)

disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "discriminator")
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")

if FLAGS.wall_clock:
    disc_train_op = train(disc_loss, FLAGS.disc_learning_rate, wall_clock, disc_vars)
    gen_train_op = train(gen_loss, FLAGS.gen_learning_rate, wall_clock, gen_vars)
else:
    disc_train_op = train(disc_loss, FLAGS.disc_learning_rate, None, disc_vars)
    gen_train_op = train(gen_loss, FLAGS.gen_learning_rate, gen_iterations, gen_vars)



# -----------------------------------------------------------------------------------
#     Run
# -----------------------------------------------------------------------------------
for nr in range(FLAGS.num_runs):
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir+'/summaries'+'/run_'+str(nr), sess.graph)
    #saver = tf.train.Saver(max_to_keep=1)

    gen_samps = []
    gen_samps.append(np.vstack([sess.run(gen_x) for _ in range(FLAGS.viz_batches)]))
    
    summary_str = sess.run(summary_op, feed_dict={gamma_plh: FLAGS.gamma})
    summary_writer.add_summary(summary_str, 0)
    
    
    for i in range(FLAGS.disc_pretrain_steps):
        sess.run(disc_train_op, feed_dict={gamma_plh: FLAGS.gamma})

    for i in tqdm(range(1, FLAGS.max_steps+1)):
        
        # ANNEALING (EXPONENTIAL DECAY gamma*decay_factor^t/T [default decay_factor=0.01])
        if FLAGS.annealing:
            gamma = FLAGS.gamma*FLAGS.decay_factor**(i/FLAGS.max_steps)
        else:
            gamma = FLAGS.gamma

        for disc_update_steps in range(FLAGS.disc_update_steps):
            sess.run(disc_train_op, feed_dict={gamma_plh: gamma})
        for gen_update_steps in range(FLAGS.gen_update_steps):
            sess.run(gen_train_op, feed_dict={gamma_plh: gamma})

        if FLAGS.wall_clock:
            current_step = tf.train.global_step(sess, wall_clock)
        else:
            current_step = tf.train.global_step(sess, gen_iterations)

        summary_str = sess.run(summary_op, feed_dict={gamma_plh: gamma})
        summary_writer.add_summary(summary_str, current_step)

        if i % FLAGS.plot_freq == 0:
            gen_samps.append(np.vstack([sess.run(gen_x) for _ in range(FLAGS.viz_batches)]))

    summary_writer.flush()


    # Save & Visualize

    np.save(log_dir+'/summaries'+'/run_'+str(nr)+'/gen_samps.npy', gen_samps)

    frames = len(gen_samps)
    fig, ax = plt.subplots(1, frames, figsize=(2*frames, 2), subplot_kw=dict(projection='3d'))

    for i, samps in enumerate(gen_samps):
        samps = np.swapaxes(samps, 0, 1)
        density = stats.gaussian_kde(samps)(samps)
        idx = density.argsort()
        x, y, z, density = samps[0, idx], samps[1, idx], samps[2, idx], density[idx]
        density-=density[0]
        density/=density[-1]
        density = [[(230*(1-d))/255, (230*(1-d)/255), (128+122*(1-d))/255] for d in density]
        ax[i].scatter(x, y, z, s=1, c=density)
        ax[i].set_xlim([-1.0,2.0])
        ax[i].set_ylim([-1.0,2.0])
        ax[i].set_zlim([-0.5,1.5])
        ax[i].xaxis.set_ticklabels([])
        ax[i].yaxis.set_ticklabels([])
        ax[i].zaxis.set_ticklabels([])
        ax[i].view_init(azim=30)
        if FLAGS.wall_clock:
            step = (i*FLAGS.plot_freq*(FLAGS.gen_update_steps+FLAGS.disc_update_steps) + np.heaviside(i, 0)*FLAGS.disc_pretrain_steps)
        else:
            step = (i*FLAGS.plot_freq*FLAGS.gen_update_steps)
        ax[i].set_title('step %d\n'%step)
    plt.savefig(log_dir+'/'+file_name+'_num_'+str(nr)+'.png', format='png', dpi=600, bbox_inches='tight')
    plt.close()

