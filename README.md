# Stabilizing_GANs
Implementation of the NIPS17 paper [Stabilizing Training of Generative Adversarial Networks through Regularization](https://arxiv.org/abs/1705.09367)


## JS-Regularizer Code

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

## GAN objective

```python
disc_loss = tf.reduce_mean( tf.nn.sigmoid_cross_entropy_with_logits(logits=D1_logits, labels=tf.ones_like(D1_logits))
                           +tf.nn.sigmoid_cross_entropy_with_logits(logits=D2_logits, labels=tf.zeros_like(D2_logits)) )

if not flags.unreg:
    disc_reg = Discriminator_Regularizer(D1_logits, data, D2_logits, G)
    disc_loss += (gamma/2.0)*disc_reg

```

A note on our notation: 

    D1 = D_real, D2 = D_fake

    gamma is a tf-placeholder that we feed with the annealed value of gamma


#Full implementation to follow soon!
