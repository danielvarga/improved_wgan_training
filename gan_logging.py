import tensorflow as tf


# TODO use keyword args
def log_weights_grads(gen_gvs, disc_gvs, params):
    for param_name, param in params.iteritems():
        print param_name, param
        tf.summary.histogram(param_name+"/weights", param)
    
    if gen_gvs is not None:
        for grad, var in gen_gvs:
            if grad is not None:
                tf.summary.histogram(var.name + "/gradients", grad)
    if disc_gvs is not None:
        for grad, var in disc_gvs:
            if grad is not None:
                tf.summary.histogram(var.name + "/gradients", grad)


def log_slopes(BATCH_SIZE, OUTPUT_DIM, ALPHA_COUNT, Generator, Discriminator, fixed_noise_samples):
    alphas = tf.placeholder(tf.float32, shape=(BATCH_SIZE, ALPHA_COUNT))
    alphas1 = tf.expand_dims(alphas, axis=-1)
    real_data_ph = tf.placeholder(tf.float32, shape=(BATCH_SIZE, OUTPUT_DIM))
    real_data_ph1 = tf.expand_dims(real_data_ph, axis=1)
    fake_data = Generator(BATCH_SIZE)
    fake_data = tf.expand_dims(fake_data, axis=1)
    
    x = alphas1*fake_data + (1-alphas1)*real_data_ph1

    alpha_to_disc_cost_op = Discriminator(x)

    grad_by_alphas = tf.gradients(alpha_to_disc_cost_op, alphas)[0]

    grad_by_x = tf.gradients(Discriminator(x), [x])[0]
    slopes_for_alphas = tf.sqrt(tf.reduce_sum(tf.square(grad_by_x), reduction_indices=[2]))

    x2 = tf.random_uniform(x.shape, minval=-1, maxval=1)
    grad_by_x2 = tf.gradients(Discriminator(x2), [x2])[0]
    slopes_for_x2 = tf.sqrt(tf.reduce_sum(tf.square(grad_by_x2), reduction_indices=[2]))
    tf.summary.histogram("slopes_at_random", slopes_for_x2)

    tf.summary.histogram("slopes_for_all_alphas", slopes_for_alphas)
    tf.summary.histogram("slopes_for_alpha0", slopes_for_alphas[:, 0])
    tf.summary.histogram("slopes_for_alpha1", slopes_for_alphas[:, -1])

    tf.summary.histogram("unidirectional_grad_at_all_alphas", grad_by_alphas)
    tf.summary.histogram("unidirectional_grad_at_alpha0", grad_by_alphas[:, 0])
    tf.summary.histogram("unidirectional_grad_at_alpha1", grad_by_alphas[:, -1])

    tf.summary.image("generated", tf.reshape(fixed_noise_samples, (128, 28, 28, 1)), max_outputs=50)

    return alphas, real_data_ph, slopes_for_alphas
