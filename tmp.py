import tensorflow as tf
import numpy as np

batch_size = 10
updates = 4000
learning_rate = 0.001
prior_mu = -2.5
prior_std = 0.5
noise_range = 5
data_shape = 507

def sample_z(size=1):
    return np.random.normal(-noise_range, noise_range, size=[size, data_shape])

graph = tf.get_default_graph()

with tf.Session(graph=graph) as sess:
    new_saver = tf.train.import_meta_graph('./logs/my_test_model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./logs/'))
    # test_z = tf.convert_to_tensor(sample_z(1))
    test_z = sample_z(2)

    # print(graph.get_name_scope())
    # input = graph.get_operation_by_name("z_p").outputs[0]
    # prediction = graph.get_operation_by_name("x_g").outputs[0]
    z_p = graph.get_tensor_by_name("z_p:0")
    op_to_restore = graph.get_tensor_by_name("out_gen:0")
    # newdata=put your data here
    print(sess.run(op_to_restore, feed_dict={z_p: test_z}).tolist())
