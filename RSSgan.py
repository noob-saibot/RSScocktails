import numpy as np
import tensorflow as tf

batch_size = 64
updates = 40000
learning_rate = 0.01
prior_mu = -2.5
prior_std = 0.5
noise_range = 5

gen_weights = {
    'w1': tf.Variable(tf.random_normal([1, 5])),
    'b1': tf.Variable(tf.random_normal([5])),
    'w2': tf.Variable(tf.random_normal([5, 1])),
    'b2': tf.Variable(tf.random_normal([1])),
}

disc_weights = {
    'w1': tf.Variable(tf.random_normal([1, 10])),
    'b1': tf.Variable(tf.random_normal([10])),
    'w2': tf.Variable(tf.random_normal([10, 10])),
    'b2': tf.Variable(tf.random_normal([10])),
    'w3': tf.Variable(tf.random_normal([10, 1])),
    'b3': tf.Variable(tf.random_normal([1])),
}

z_p = tf.placeholder('float', [None, 1])
x_d = tf.placeholder('float', [None, 1])
g_h = tf.nn.softplus(tf.add(tf.matmul(z_p, gen_weights['w1']), gen_weights['b1']))
x_g = tf.add(tf.matmul(g_h, gen_weights['w2']), gen_weights['b2'])

# Add summary ops to collect data
h1gw = tf.summary.histogram("weights_g1", gen_weights['w1'])
h1gb = tf.summary.histogram("biases_g1", gen_weights['b1'])
h2gw = tf.summary.histogram("weights_g2", gen_weights['w2'])
h2gb = tf.summary.histogram("biases_g2", gen_weights['b2'])

h1dw = tf.summary.histogram("weights_d1", disc_weights['w1'])
h1db = tf.summary.histogram("biases_d1", disc_weights['b1'])
h2dw = tf.summary.histogram("weights_d2", disc_weights['w2'])
h2db = tf.summary.histogram("biases_d2", disc_weights['b2'])
h3dw = tf.summary.histogram("weights_d3", disc_weights['w2'])
h3db = tf.summary.histogram("biases_d3", disc_weights['b2'])

def discriminator(x):
    d_h1 = tf.nn.tanh(tf.add(tf.matmul(x, disc_weights['w1']), disc_weights['b1']))
    d_h2 = tf.nn.tanh(tf.add(tf.matmul(d_h1, disc_weights['w2']), disc_weights['b2']))
    score = tf.nn.sigmoid(tf.add(tf.matmul(d_h2, disc_weights['w3']), disc_weights['b3']))
    return score

x_data_score = discriminator(x_d)
x_gen_score = discriminator(x_g)

D_cost = -tf.reduce_mean(tf.log(x_data_score) + tf.log(1.0 - x_gen_score))
G_cost = tf.reduce_mean(tf.log(1.0 - x_gen_score))

tf.summary.scalar("cost_function_G", G_cost)
tf.summary.scalar("cost_function_D", D_cost)

optimizer = tf.train.GradientDescentOptimizer(learning_rate)
D_optimizer = optimizer.minimize(D_cost, var_list=list(disc_weights.values()))
G_optimizer = optimizer.minimize(G_cost, var_list=list(gen_weights.values()))


def sample_z(size=batch_size):
    return np.random.uniform(-noise_range, noise_range, size=[size, 1])


def sample_x(size=batch_size, mu=prior_mu, std=prior_std):
    return np.random.normal(mu, std, size=[size, 1])

merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

summary_writer = tf.summary.FileWriter('logs', graph_def=tf.GraphDef())

for i in range(updates):
    z_batch = sample_z()
    x_batch = sample_x()
    sess.run(D_optimizer, feed_dict={z_p: z_batch, x_d: x_batch})
    z_batch = sample_z()
    sess.run(G_optimizer, feed_dict={z_p: z_batch})

    summary_str = sess.run(merged_summary_op, feed_dict={z_p: z_batch, x_d: x_batch})
    summary_writer.add_summary(summary_str, i * batch_size + i)

