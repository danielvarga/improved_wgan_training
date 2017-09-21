import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns

# input: discriminator and a set of points
# find the greatest slopes in the neighbourhood of the points
# return the greatest slopes and the corresponding points
def find_greatest_slopes(discriminator, images, iters, lr, session):
    input_ph = tf.placeholder(tf.float32, shape=images.shape)
    output = discriminator(input_ph)
    gradients = tf.gradients(output, [input_ph])[0] 
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    slope_gradients = tf.gradients(slopes, [input_ph])[0]

    for iteration in xrange(iters):
        _slope_gradients = session.run([slope_gradients],feed_dict={input_ph: images})[0]
        images += lr * _slope_gradients
        images = np.clip(images, a_min=0.0, a_max=1.0)
    _slopes, _output = session.run([slopes, output],feed_dict={input_ph: images})
    return images, _slopes, _output

# create joint plot with marginals
def scatterWithMarginals(xs, ys, title, filename):
    d = {"x": xs, "y": ys}
    df = pd.DataFrame(data=d)
    sns_plot = sns.jointplot(x="x", y="y", data=df, kind="kde")
    sns_plot.fig.suptitle(title)
    sns_plot.savefig(filename)


 #    # weight regularization
 # -    if WEIGHT_DECAY_FACTOR > 0:
 # -
 # -        with tf.variable_scope('weights_norm') as scope:
 # -            weight_loss = tf.reduce_sum(
 # -                input_tensor = WEIGHT_DECAY_FACTOR*tf.stack(
 # -                    [tf.nn.l2_loss(tf.maximum(0.01, var)) for var in disc_filters]
 # -                ),
 # -                name='weight_loss'
 # -            )
 # -        disc_cost += weight_loss
