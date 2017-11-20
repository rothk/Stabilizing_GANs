# Stabilizing_GANs

Implementation of the NIPS17 paper [Stabilizing Training of Generative Adversarial Networks through Regularization](https://arxiv.org/abs/1705.09367).

## Summary



Check also Ferenc Huszar's excellent blog post on the topic [From Instance Noise to Gradient Regularisation](http://www.inference.vc/from-instance-noise-to-gradient-regularisation/).


Big thanks to Ishaan Gulrajani [(igul222)](https://github.com/igul222/improved_wgan_training), Taehoon Kim [(carpedm20)](https://github.com/carpedm20/DCGAN-tensorflow) and Ben Poole [(poolio)](https://github.com/poolio/unrolled_gan) for their open-source GAN implementations, on which our modified code is based. 




## JS-Regularizer & Regularized GAN Objective:

Simply copy-paste this definition into your GAN code and modify the `disc_loss += (gamma/2.0)*disc_reg` (it's _plus_ the regularizer because the maximization of the discriminator's objective is implemented as a minimization)

```python
def Discriminator_Regularizer(D1_logits, D1_arg, D2_logits, D2_arg):
    D1 = tf.nn.sigmoid(D1_logits)
    D2 = tf.nn.sigmoid(D2_logits)
    grad_D1_logits = tf.gradients(D1_logits, D1_arg)[0]
    grad_D2_logits = tf.gradients(D2_logits, D2_arg)[0]
    grad_D1_logits_norm = tf.norm(tf.reshape(grad_D1_logits, [batch_size,-1]), axis=1, keep_dims=True)
    grad_D2_logits_norm = tf.norm(tf.reshape(grad_D2_logits, [batch_size,-1]), axis=1, keep_dims=True)

    #set keep_dims=True/False such that grad_D_logits_norm.shape == D.shape
    assert grad_D1_logits_norm.shape == D1.shape
    assert grad_D2_logits_norm.shape == D2.shape

    reg_D1 = tf.multiply(tf.square(1.0-D1), tf.square(grad_D1_logits_norm))
    reg_D2 = tf.multiply(tf.square(D2), tf.square(grad_D2_logits_norm))
    disc_regularizer = tf.reduce_mean(reg_D1 + reg_D2)
    return disc_regularizer
```
 
**Regularized GAN Objective:**

```python
disc_loss  = tf.reduce_mean( 
     tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_logits, labels=tf.ones_like(D1_logits))
    +tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logits, labels=tf.zeros_like(D2_logits)) )

disc_reg   = Discriminator_Regularizer(D1_logits, data, D2_logits, G)
disc_loss += (gamma/2.0)*disc_reg

gen_loss   = tf.reduce_mean(
     tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logits, labels=tf.ones_like(D2_logits)) )

```

**Notation:**

- `D1 = disc_real, D2 = disc_fake`
- `gamma` is a placeholder that is fed with the annealed value of gamma during training




## Prerequisites

- Python3+
- TensorFlow, NumPy, SciPy, Matplotlib, tqdm


## Usage

**carpedm20_DCGAN**:

    $ python3 main.py --gamma=2.0 --annealing --epochs=10 --dataset=celebA --output_height=32

**igul222_GANs**:

    $ python3 gan_64x64.py --architecture=ResNet --gamma=2.0 --annealing --iters=100000 --dataset=lsun

**3DGAN**:

    $ python3 3DGAN.py --gamma=0.1 --max_steps=100000

**Loading data**:

For **carpedm20_DCGAN** the datasets are by default loaded from `./data/dataset`.

For **igul222_GANs** check the `load_*.py` files in `/tflib` for the appropriate directory naming `~/data/dataset/train` resp `~/data/dataset/eval` to load the training and test data from and adapt lines 50-60 in `gan_64x64.py`.

## Datasets

CelebA, LSUN & MNIST datasets can be downloaded with carpedm20's `download.py`:

    $ python download.py celebA lsun mnist

Cifar10 & ImageNet datasets can be downloaded from 

    $ https://www.cs.toronto.edu/~kriz/cifar.html
    $ http://image-net.org/small/download.php





