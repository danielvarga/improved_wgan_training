import tensorflow as tf
import util

def calculate_losses(SUB_BATCH_SIZE, real_data, Generator, Discriminator, MODE, alpha_strategy, LAMBDA, aggregator = tf.reduce_max, WEIGHT_DECAY=0.0, params_for_wd=None, remember_last_activation=False):
            fake_data = Generator(SUB_BATCH_SIZE)

            disc_real = Discriminator(real_data)
            disc_fake = Discriminator(fake_data)

            if MODE == 'wgan':
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

            elif MODE in ('wgan-gp', 'wgan-gs') :
                gen_cost = -tf.reduce_mean(disc_fake)
                disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

                print "alpha_strategy", alpha_strategy
                interpolates = get_slope_samples(real_data, fake_data, alpha_strategy, SUB_BATCH_SIZE)
                
                if remember_last_activation:
                            gradients = tf.gradients(Discriminator(interpolates, remember_last_activation=remember_last_activation), [interpolates])[0]
                else:
                            gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]

                initial_slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                final_slopes = initial_slopes

                # Original WGAN-GP penalty:
                gradient_penalty = tf.reduce_mean((initial_slopes-1.)**2)

                # Flat
                # TODO This could be tf.reduce_mean(tf.maximum(1., slopes*slopes) - 1.0), try it.
                # gradient_penalty = tf.reduce_mean((tf.maximum(1., tf.abs(slopes)) - 1.0)**2)

                # L2
                # it's simply tf.square(slopes), but I'm not sure whether TF optimizes away tf.square(tf.sqrt()).
                # gradient_penalty = tf.reduce_sum(tf.square(gradients), reduction_indices=[1])

                if MODE == 'wgan-gs':
                    print "gradient shrinking", aggregator
                    norm_factor = aggregator(initial_slopes)
                    disc_real /= norm_factor
                    disc_fake /= norm_factor
                    final_slopes /= norm_factor
                    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
                    gen_cost = -tf.reduce_mean(disc_fake)
                else:
                    if LAMBDA == 0:
                        print "are you sure you want a LAMBDA=0 wgan-gp?"
                    disc_cost += LAMBDA * gradient_penalty
            
            if WEIGHT_DECAY > 0:
                        print "Applying weight decay: ", WEIGHT_DECAY
                        disc_cost += util.get_weight_loss(params_for_wd, WEIGHT_DECAY)

            return gen_cost, disc_cost, initial_slopes, final_slopes, gradient_penalty

def get_slope_samples(real_data, fake_data, alpha_strategy, SUB_BATCH_SIZE):
            if alpha_strategy == "uniform":
                        # Original WGAN-GP interpolation:
                        alpha = tf.random_uniform(
                                    shape=[SUB_BATCH_SIZE,1], 
                                    minval=0.,
                                    maxval=1.
                        )
            elif alpha_strategy == "bernoulli":
                        alpha = tf.where(alpha < 0.5, tf.ones([SUB_BATCH_SIZE,1]), tf.zeros([SUB_BATCH_SIZE, 1]))
            elif alpha_strategy == "fake":
                        # Straight GP1:
                        alpha = tf.ones([SUB_BATCH_SIZE,1])
            elif alpha_strategy == "real":
                        alpha = tf.zeros([SUB_BATCH_SIZE,1])
            elif alpha_strategy == "random":
                        return tf.random_uniform(shape=fake_data.shape, minval=0.0, maxval=1.0)
            elif alpha_strategy == "real_plus_noise":
                        alpha = tf.zeros([SUB_BATCH_SIZE,1])

            differences = fake_data - real_data
            interpolates = real_data + (alpha*differences)
            if alpha_strategy == "real_plus_noise":
                        interpolates += tf.random_normal(shape=fake_data.shape, stddev=0.1)

            return interpolates
