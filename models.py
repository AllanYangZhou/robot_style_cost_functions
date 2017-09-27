import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras import backend as K
import numpy as np

import utils
import constants as c


class MLP:
    def __init__(self, input_dim):
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=input_dim, activation='tanh'))

        self.model.add(Dropout(0.5))
        self.model.add(Dense(16, activation='tanh'))

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))


    def __call__(self, x):
        return self.model(x)


class Recurrent:
    def __init__(self, input_dim):
        self.model = Sequential()
        self.model.add(TimeDistributed(Dense(32, activation='tanh'), input_shape=(10, input_dim)))
        self.model.add(TimeDistributed(Dropout(0.5)))

        self.model.add(LSTM(16, dropout=0.5, return_sequences=True))

        self.model.add(Flatten())
        self.model.add(Dense(1))


    def __call__(self, x):
        return self.model(x)


class CostFunction:
    def __init__(self,
                 robot,
                 load_path=None,
                 num_wps=10,
                 num_dofs=10,
                 recurrent=False,
                 use_total_duration=False,
                 normalize=False):
        self.use_total_duration = use_total_duration
        self.robot = robot
        self.env = robot.GetEnv()
        self.sess = tf.Session()
        self.trajA_ph = tf.placeholder(tf.float32,
                                       shape=[None, num_wps, num_dofs],
                                       name='trajA_ph')
        self.trajB_ph = tf.placeholder(tf.float32,
                                       shape=[None, num_wps, num_dofs],
                                       name='trajB_ph')
        if normalize:
            tf_world_feature_min = tf.constant(c.world_feature_min.astype(np.float32))
            tf_world_frange = tf.constant(c.world_frange.astype(np.float32))
            trajA = (2 * (self.trajA_ph - tf_world_feature_min) / (tf_world_frange)) - 1.
            trajB = (2 * (self.trajB_ph - tf_world_feature_min) / (tf_world_frange)) - 1.
        else:
            trajA = self.trajA_ph
            trajB = self.trajB_ph
        self.label_ph = tf.placeholder(tf.int32, shape=[None], name='label_ph')
        self.cost_label_ph = tf.placeholder(tf.float32, shape=[None], name='cost_label_ph')
        self.gan_label_ph = tf.placeholder(tf.float32, shape=[None], name='gan_label_ph')

        batch_size = tf.shape(trajA)[0]
        if recurrent:
            self.mlp = Recurrent(
                num_dofs + int(use_total_duration))
        else:
            self.mlp =  MLP(2*num_dofs + 1 + int(use_total_duration))
        if use_total_duration:
            self.timeA_ph = tf.placeholder(tf.float32, shape=[None, 1])
            self.timeB_ph = tf.placeholder(tf.float32, shape=[None, 1])
            trajA = tf.concat([trajA, self.timeA_ph], axis=1)
            trajB = tf.concat([trajB, self.timeB_ph], axis=1)
        if recurrent:
            self.costA, self.costB = self.mlp(trajA), self.mlp(trajB)
        else:
            costA, costB = [], []
            for i in range(num_wps):
                prev_idx = max(i-1, 0)
                currA_state = trajA[:,i,:]
                velA = trajA[:,prev_idx,:] - currA_state
                currB_state = trajB[:,i,:]
                velB = trajB[:,prev_idx,:] - currB_state
                wp_num = tf.fill([batch_size, 1], float(i))
                costA.append(self.mlp(tf.concat([currA_state, velA, wp_num], axis=1)))
                costB.append(self.mlp(tf.concat([currB_state, velB, wp_num], axis=1)))
            self.costA, self.costB = tf.add_n(costA), tf.add_n(costB)

        # Probability proportional to e^{-cost}
        cost_logits = -1 * tf.concat([self.costA, self.costB], axis=-1)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=cost_logits, labels=self.label_ph))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.cost_loss = tf.reduce_mean(tf.square(self.costA - self.cost_label_ph))
        self.cost_train_op = tf.train.AdamOptimizer().minimize(self.cost_loss)
        self.gan_cost_loss = self.gan_label_ph * self.costA
        self.gan_train_op = tf.train.AdamOptimizer().minimize(self.gan_cost_loss)


        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())
        self.sess.run(init_op)
        if load_path:
            self.load_model(load_path)
        self.num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])


    def cost_traj(self, waypoints, total_time=0):
        '''Returns cost for a single trajectory with shape (num_wps, 7).'''
        fd = {
            self.trajA_ph: waypoints[None],
            K.learning_phase(): False
        }
        if self.use_total_duration:
            fd[self.timeA_ph] = [[total_time]]
        cost = self.sess.run(self.costA, feed_dict=fd)
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


    def train(self, trajsA, trajsB, labels, total_time=(0,0)):
        fd = {
            self.trajA_ph: trajsA,
            self.trajB_ph: trajsB,
            self.label_ph: labels,
            K.learning_phase(): True
        }
        if self.use_total_duration:
            fd[self.timeA_ph] = [[total_time[0]]]
            fd[self.timeB_ph] = [[total_time[1]]]
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict=fd)
        return loss


    def train_cost(self, trajs, cost_labels):
        cost_loss, _ = self.sess.run([self.cost_loss, self.cost_train_op], feed_dict={
            self.trajA_ph: trajs,
            self.cost_label_ph: cost_labels,
            K.learning_phase(): True
        })
        return cost_loss


    def train_gan_cost(self, traj, gan_labels):
        self.sess.run([self.gan_cost_loss], feed_dict={
            self.trajA_ph: trajs,
            self.gan_label_ph: gan_labels,
            K.learning_phase(): True
        })


    def get_cost_loss(self, trajs, cost_labels):
        cost_loss = self.sess.run(self.cost_loss, feed_dict={
            self.trajA_ph: trajs,
            self.cost_label_ph: cost_labels,
            K.learning_phase(): False
        })
        return cost_loss


    def save_model(self, path, step=None):
        if step:
            self.saver.save(self.sess, path, global_step=step)
        else:
            self.saver.save(self.sess, path)


    def load_model(self, path):
        self.saver.restore(self.sess, path)


    def naive_time_opt(self, waypoints, return_costs=False):
        with self.env:
            wf = utils.world_space_featurizer(self.robot, waypoints)
        # test_inputs = []
        # for i in range(0, 30):
            # times = .1 * i * np.ones((10, 1))
            # test_inputs.append(np.concatenate([times, wf], axis=1))
        # costs = self.cost_traj_batch(np.stack(test_inputs))
        costs = np.array([self.cost_traj(wf, total_time=(2 * (i-1) / 29.)-1)
                          for i in range(1, 31)])
        return costs if return_costs else np.argmin(costs)
