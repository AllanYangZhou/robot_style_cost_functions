import tensorflow as tf
from keras.layers import Dense, Dropout, LeakyReLU
from keras.layers.convolutional import Conv1D
from keras.models import Sequential
from keras import backend as K
import numpy as np

import utils


class MLP:
    def __init__(self, input_dim, h_size=64):
        self.model = Sequential()
        self.model.add(Dense(h_size, input_dim=input_dim, activation='tanh'))

        self.model.add(Dropout(0.5))
        self.model.add(Dense(h_size, activation='tanh'))

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))


    def __call__(self, x):
        return self.model(x)


class CostFunction:
    def __init__(self, robot, load_path=None, num_wps=10, num_dofs=10, h_size=64):
        self.robot = robot
        self.env = robot.GetEnv()
        self.sess = tf.Session()
        self.trajA_ph = tf.placeholder(tf.float32, shape=[None, num_wps, num_dofs], name='trajA_ph')
        self.trajB_ph = tf.placeholder(tf.float32, shape=[None, num_wps, num_dofs], name='trajB_ph')
        self.label_ph = tf.placeholder(tf.int32, shape=[None], name='label_ph')
        self.cost_label_ph = tf.placeholder(tf.float32, shape=[None], name='cost_label_ph')

        batch_size = tf.shape(self.trajA_ph)[0]
        input_dim = num_dofs * num_wps
        self.mlp = MLP(input_dim, h_size)
        trajA = tf.reshape(self.trajA_ph, [batch_size, input_dim])
        trajB = tf.reshape(self.trajB_ph, [batch_size, input_dim])
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


    def naive_time_opt(self, waypoints, return_costs=False):
        with self.env:
            wf = utils.world_space_featurizer(self.robot, waypoints)
        test_inputs = []
        for i in range(0, 30):
            times = .1 * i * np.ones((10, 1))
            test_inputs.append(np.concatenate([times, wf], axis=1))
        costs = self.cost_traj_batch(np.stack(test_inputs))
        return costs if return_costs else np.argmin(costs)

    def get_trajopt_cost(self):
        def cf_cost(x):
            x = x.reshape((10,7))
            f = utils.world_space_featurizer(robot, x)
            f = np.concatenate([np.ones((10,1)), f], axis=1)
            score = self.cost_traj(f)
            return score
        return cf_cost

