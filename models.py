import tensorflow as tf
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers.convolutional import Conv1D
from keras.models import Sequential
from keras import backend as K
import numpy as np


class MLP:
    def __init__(self, input_dim, h_size=64):
        self.model = Sequential()
        self.model.add(Dense(h_size, input_dim=input_dim, activation='sigmoid'))

        self.model.add(Dropout(0.5))
        self.model.add(Dense(h_size, activation='sigmoid'))

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

    def __call__(self, x):
        return self.model(x)


class CostFunction:
    def __init__(self, load_path=None, num_wps=10, num_dofs=10, h_size=64, add_total_time=False):
        self.add_total_time = add_total_time
        self.sess = tf.Session()
        self.trajA_ph = tf.placeholder(tf.float32, shape=[None, num_wps, num_dofs], name='trajA_ph')
        self.trajB_ph = tf.placeholder(tf.float32, shape=[None, num_wps, num_dofs], name='trajB_ph')
        self.label_ph = tf.placeholder(tf.int32, shape=[None], name='label_ph')
        self.cost_label_ph = tf.placeholder(tf.float32, shape=[None], name='cost_label_ph')

        batch_size = tf.shape(self.trajA_ph)[0]
        input_dim = num_dofs * num_wps
        self.mlp = MLP(input_dim + int(add_total_time), h_size)
        trajA = tf.reshape(self.trajA_ph, [batch_size, input_dim])
        trajB = tf.reshape(self.trajB_ph, [batch_size, input_dim])
        if add_total_time:
            self.trajA_time_ph = tf.placeholder(tf.float32, shape=[None, 1], name='trajA_time_ph')
            self.trajB_time_ph = tf.placeholder(tf.float32, shape=[None, 1], name='trajB_time_ph')
            trajA = tf.concat([trajA, self.trajA_time_ph], axis=1)
            trajB = tf.concat([trajB, self.trajB_time_ph], axis=1)
        self.costA, self.costB = self.mlp(trajA), self.mlp(trajB)

        # Probability proportional to e^{-cost}
        cost_logits = -1 * tf.concat([self.costA, self.costB], axis=-1)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=cost_logits, labels=self.label_ph))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.cost_loss = tf.reduce_mean(tf.square(self.costA - self.cost_label_ph))
        self.cost_train_op = tf.train.AdamOptimizer().minimize(self.cost_loss)


        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())
        self.sess.run(init_op)
        if load_path:
            self.load_model(load_path)
        self.num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])

    def cost_traj(self, waypoints):
        '''Returns cost for a single trajectory with shape (num_wps, 7).'''
        cost = self.sess.run(self.costA, feed_dict={
            self.trajA_ph: waypoints[None],
            K.learning_phase(): False
        })
        return np.squeeze(cost)

    def cost_traj_batch(self, trajs):
        '''Returns cost for a batch of trajs with shape (num_trajs, num_wps, 7).'''
        cost = self.sess.run(self.costA, feed_dict={
            self.trajA_ph: trajs,
            K.learning_phase(): False
        })
        return np.squeeze(cost)

    def corrcoef(self, test_data, labels):
        '''Correlation between model predictions and labels.'''
        return np.corrcoef(self.cost_traj_batch(test_data), labels)[0,1]

    def train(self, trajsA, trajsB, labels, total_time=(1,1)):
        fd = {
            self.trajA_ph: trajsA,
            self.trajB_ph: trajsB,
            self.label_ph: labels,
            K.learning_phase(): True
        }
        if self.add_total_time:
            fd[self.trajA_time_ph] = total_time[0]
            fd[self.trajB_time_ph] = total_time[1]
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=fd)
        return loss

    def pretrain_on_prior_cost(self, trajs, cost_labels):
        cost_loss, _ = self.sess.run([self.cost_loss, self.cost_train_op], feed_dict={
            self.trajA_ph: trajs,
            self.cost_label_ph: cost_labels,
            K.learning_phase(): True
        })
        return cost_loss

    def get_cost_loss(self, trajs, cost_labels):
        cost_loss = self.sess.run(self.cost_loss, feed_dict={
            self.trajA_ph: trajs,
            self.cost_label_ph: cost_labels,
            K.learning_phase(): False
        })
        return cost_loss

    def save_model(self, path):
        save_path = self.saver.save(self.sess, path)

    def load_model(self, path):
        self.saver.restore(self.sess, path)
