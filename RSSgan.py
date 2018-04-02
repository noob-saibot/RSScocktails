import numpy as np
import tensorflow as tf
import seaborn
import matplotlib.pyplot as plt
from RSSdb import Connector
import os

path = 'logs/'
for i in os.listdir(path):
    os.remove(path+i)


con = Connector()
df = con.get_values('proc_table')

data = df.drop('name', axis=1).values


batch_size = 20
updates = 40000
learning_rate = 0.01
prior_mu = -2.5
prior_std = 0.5
noise_range = 5


gen_weights = {
    'w1': tf.get_variable("gw1", shape=[data.shape[1], 30],
                          initializer=tf.keras.initializers.he_normal()),
    'b1': tf.Variable(tf.random_normal([1])),
    'w2': tf.get_variable("gw2", shape=[30, 10],
                          initializer=tf.keras.initializers.he_normal()),
    'b2': tf.Variable(tf.random_normal([1])),
    'w3': tf.get_variable("gw3", shape=[10, data.shape[1]],
                          initializer=tf.keras.initializers.he_normal()),
    'b3': tf.Variable(tf.random_normal([1])),
}

disc_weights = {
    'w1': tf.get_variable("dw1", shape=[data.shape[1], 3],
                          initializer=tf.contrib.layers.xavier_initializer()),
    'b1': tf.Variable(tf.random_normal([1])),
    'w2': tf.get_variable("dw2", shape=[3, 5],
                          initializer=tf.contrib.layers.xavier_initializer()),
    'b2': tf.Variable(tf.random_normal([1])),
    'w3': tf.get_variable("dw3", shape=[5, data.shape[1]],
                          initializer=tf.contrib.layers.xavier_initializer()),
    'b3': tf.Variable(tf.random_normal([1])),
}


z_p = tf.placeholder('float', [None, data.shape[1]], name='z_p')
x_d = tf.placeholder('float', [None, data.shape[1]])
g_h1 = tf.nn.leaky_relu(tf.add(tf.matmul(z_p, gen_weights['w1']), gen_weights['b1']))
g_h2 = tf.nn.leaky_relu(tf.add(tf.matmul(g_h1, gen_weights['w2']), gen_weights['b2']))

# batch normalization
scale1 = tf.Variable(tf.ones([10]))
beta1 = tf.Variable(tf.zeros([10]))
batch_mean2, batch_var2 = tf.nn.moments(g_h2,[0])
scale2 = tf.Variable(tf.ones([10]))
beta2 = tf.Variable(tf.zeros([10]))
epsilon = 1e-3

g_h2 = tf.nn.batch_normalization(g_h2, batch_mean2,batch_var2,beta2,scale2,epsilon)
x_g = tf.nn.leaky_relu(tf.add(tf.matmul(g_h2, gen_weights['w3']), gen_weights['b3'], name='out_gen'))

h1gw = tf.summary.histogram("weights_g1", gen_weights['w1'])
h1gb = tf.summary.histogram("biases_g1", gen_weights['b1'])
h2gw = tf.summary.histogram("weights_g2", gen_weights['w2'])
h2gb = tf.summary.histogram("biases_g2", gen_weights['b2'])
# h3gw = tf.summary.histogram("weights_g3", gen_weights['w3'])
# h3gb = tf.summary.histogram("biases_g3", gen_weights['b3'])

h1dw = tf.summary.histogram("weights_d1", disc_weights['w1'])
h1db = tf.summary.histogram("biases_d1", disc_weights['b1'])
h2dw = tf.summary.histogram("weights_d2", disc_weights['w2'])
h2db = tf.summary.histogram("biases_d2", disc_weights['b2'])
h3dw = tf.summary.histogram("weights_d3", disc_weights['w3'])
h3db = tf.summary.histogram("biases_d3", disc_weights['b3'])


def discriminator(x):
    d_h1 = tf.nn.tanh(tf.add(tf.matmul(x, disc_weights['w1']), disc_weights['b1']))
    d_h2 = tf.nn.tanh(tf.add(tf.matmul(d_h1, disc_weights['w2']), disc_weights['b2']))

    logits = tf.add(tf.matmul(d_h2, disc_weights['w3']), disc_weights['b3'])
    return logits


x_data_score = discriminator(x_d)
x_gen_score = discriminator(x_g)


D_plus_cost = tf.reduce_mean(tf.nn.relu(x_data_score)
                             - x_data_score
                             + tf.log(1.0 + tf.exp(-tf.abs(x_data_score))))


D_minus_cost = tf.reduce_mean(tf.nn.relu(x_gen_score)
                              + tf.log(1.0 + tf.exp(-tf.abs(x_gen_score))))

G_cost = tf.reduce_mean(tf.nn.relu(x_gen_score)
                        - x_gen_score
                        + tf.log(1.0 + tf.exp(-tf.abs(x_gen_score))))

D_cost = D_plus_cost + D_minus_cost

tf.summary.scalar("cost_function_G", G_cost)
tf.summary.scalar("cost_function_D", D_cost)


optimizer = tf.train.AdagradOptimizer(learning_rate)
D_optimizer = optimizer.minimize(D_cost, var_list=list(disc_weights.values()))
G_optimizer = optimizer.minimize(G_cost, var_list=list(gen_weights.values()))

def sample_z(size=batch_size):
    return np.random.normal(-noise_range, noise_range, size=[size, data.shape[1]])

def sample_x(size=batch_size, mu=prior_mu, std=prior_std):
    return np.random.normal(mu, std, size=[size, 1])

def sample_data(data, size=batch_size):
    idx = np.random.choice(len(data), size)
    return data[idx]

merged_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session(config=tf.ConfigProto())
sess.run(init)

summary_writer = tf.summary.FileWriter('logs', graph_def=tf.GraphDef(), filename_suffix='rs')

state = True
counter = 0
for i in range(updates):
    z_batch = sample_z()
    x_batch = sample_data(data)

    # Forced slow down updates of discriminator
    # counter += 1
    # if counter == 2:
    #     test_z = sample_z(10)
    #     test_x = sample_data(data, 10)
    #     cost1 = sess.run(D_cost, feed_dict={z_p: test_z, x_d: test_x})
    #     if cost1 > 1.2:
    #         sess.run(D_optimizer, feed_dict={z_p: z_batch, x_d: x_batch})
    #     counter = 0
    sess.run(D_optimizer, feed_dict={z_p: z_batch, x_d: x_batch})

    z_batch = sample_z()
    sess.run(G_optimizer, feed_dict={z_p: z_batch})

    if i % 5000 == 0:
        print('Step: ', i)
        test_z = sample_z(5000)
        test_x = sample_data(data, 5000)

        cost1 = sess.run(D_cost, feed_dict={z_p: test_z, x_d: test_x})
        cost2 = sess.run(G_cost, feed_dict={z_p: test_z})
        print('Discriminator cost: ', cost1)
        print('Generator cost: ', cost2)
        print('Subtract:', cost1 - cost2)

    summary_str = sess.run(merged_summary_op, feed_dict={z_p: z_batch, x_d: x_batch})
    summary_writer.add_summary(summary_str, i * batch_size + i)
saver.save(sess, './logs/my_test_model')
del sess

