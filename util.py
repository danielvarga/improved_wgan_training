import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops

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
    sns_plot = sns.jointplot(x="x", y="y", data=df, kind="hex")
    sns_plot.fig.suptitle(title)
    sns_plot.savefig(filename)

def jacobian(y_flat, x):
    n = y_flat.shape[0]

    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]

    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[j], x))),
        loop_vars)
    result = jacobian.stack()
    return result

def jacobian_by_batch(ys, xs):
    n = ys.shape[1]

    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]

    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(ys[:,j], xs))),
        loop_vars)
    result = jacobian.stack()
    result = result[:,0]
    result = tf.transpose(result, perm=[1,0,2])
    return result

def get_weight_loss(weights, wd):
    losses = [tf.nn.l2_loss(var) for var in weights]
    weight_loss = wd * tf.reduce_sum(losses)
    return weight_loss

    # with tf.variable_scope('weights_norm') as scope:
    #     weight_loss = tf.reduce_sum(
    #         input_tensor = WEIGHT_DECAY_FACTOR*tf.stack(
    #             [tf.nn.l2_loss(tf.maximum(var for var in disc_filters]
    #         ),
    #         name='weight_loss'
    #     )
    # return weight_loss

    #   scope = tf.variable_scope('weights_norm')

# implement piecewise linear (learning rate) schedule
def piecewise_linear_schedule(global_step, boundaries, values, name=None):
    if global_step is None:
        raise ValueError("global_step is required for piecewise_linear_schedule.")
    assert len(boundaries) == len(values), "boundaries length ({}) should equal values length ({})".format(len(boundaries), len(values))

    with ops.name_scope(name, "piecewise_linear_schedule",
                      [global_step, boundaries, values]) as name:
        x = math_ops.cast(global_step, tf.float32)
        pred_fn_pairs = []
        pred_fn_pairs.append((x <= boundaries[0], lambda: values[0]))
        pred_fn_pairs.append((x > boundaries[-1], lambda: values[-1]))
        for low, high, low_v, high_v in zip(boundaries[:-1], boundaries[1:], values[:-1], values[1:]):
            # Need to bind v here; can do this with lambda v=v: ...
            pred = (x > low) & (x <= high)
            r = (x - low) / (high - low)
            v = r * high_v + (1-r) * low_v
            pred_fn_pairs.append((pred, lambda v=v: v))

        # The default isn't needed here because our conditions are mutually
        # exclusive and exhaustive, but tf.case requires it.
        default = lambda: values[0]
        return control_flow_ops.case(pred_fn_pairs, default, exclusive=True)
