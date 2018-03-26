import pickle
import pandas as pd
import tensorflow as tf
import numpy as np


class RSSautoencoder:
    def __init__(self):
        self.data = self._load_test()

    @property
    def config(self):
        return {'batch_size': 10,
                'latent_space': 128,
                'learning_rate': 0.1}

    @property
    def weights(self):
        return {'conv': tf.Variable(tf.truncated_normal([1, 5, 2, 4], stddev=0.1)),
                'a_hidden': tf.Variable(tf.truncated_normal([4], stddev=0.1)),
                'deconv': tf.Variable(tf.truncated_normal([5, 5, 2, 4], stddev=0.1)),
                'a_visible': tf.Variable(tf.truncated_normal([2], stddev=0.1)),
                }

    def _load_test(self):
        with open('data_encoder', 'rb') as file:
            return pickle.load(file)

    def create_network(self, data):
        input_shape = tf.stack([self.config['batch_size'], 1, 6211, 2])
        ae_input = tf.placeholder(tf.float32, [self.config['batch_size'], 6211])
        images = tf.reshape(ae_input, [-1, 1, 6211, 2])
        hidden_logits = (tf.nn.conv2d(ae_input, self.weights['conv'], strides=[1, 2, 2, 1], padding='SAME')
                         + self.weights['a_hidden'])

        hidden = tf.nn.sigmoid(hidden_logits)

        visible_logits = tf.nn.conv2d_transpose(hidden, self.weights['deconv'], input_shape, strides=[1,2,2,1],
                                                padding='SAME') + self.weights['a_visible']

        visible = tf.nn.sigmoid(visible_logits)

        optimizer = tf.train.AdagradOptimizer(self.config['learning_rate'])

        conv_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(hidden_logits, images))
        conv_op = optimizer.minimize(conv_cost)


        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        for i in range(10000):
            x_batch = data
            sess.run(conv_op, feed_dict={ae_input: x_batch})


if __name__ == '__main__':
    R = RSSautoencoder()
    cos = R.data['cosine']
    euc = R.data['euclidean']
    tdm = np.stack([cos.values, euc.values], axis=1)
    print(tdm.shape)
    exit()
    R.create_network(tdm)
