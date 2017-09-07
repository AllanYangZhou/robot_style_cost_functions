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
        #self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(h_size, activation='sigmoid'))
        #self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

    def __call__(self, x):
        return self.model(x)


class CostFunction:
    def __init__(self, load_path=None, num_wps=10, num_dofs=10, per_waypoint=False, h_size=64):
        self.sess = tf.Session()
        self.waypoint_ph = tf.placeholder(tf.float32, shape=[None, num_dofs], name='waypoint_ph')
        self.trajA_ph = tf.placeholder(tf.float32, shape=[None, num_wps, num_dofs], name='trajA_ph')
        self.trajB_ph = tf.placeholder(tf.float32, shape=[None, num_wps, num_dofs], name='trajB_ph')
        self.label_ph = tf.placeholder(tf.int32, shape=[None], name='label_ph')
        self.cost_label_ph = tf.placeholder(tf.float32, shape=[None], name='cost_label_ph')

        batch_size = tf.shape(self.trajA_ph)[0]
        input_dim = num_dofs if per_waypoint else num_dofs * num_wps
        self.mlp = MLP(input_dim, h_size)
        self.waypoint_cost = self.mlp(self.waypoint_ph) if per_waypoint else None
        if per_waypoint:
            A_costs, B_costs = [], []
            for wp in range(num_wps):
                A_costs.append(self.mlp(self.trajA_ph[:,wp,:]))
                B_costs.append(self.mlp(self.trajB_ph[:,wp,:]))
            self.costA = tf.add_n(A_costs)
            self.costB = tf.add_n(B_costs)
        else:
            trajA = tf.reshape(self.trajA_ph, [batch_size, input_dim])
            trajB = tf.reshape(self.trajB_ph, [batch_size, input_dim])
            self.costA, self.costB = self.mlp(trajA), self.mlp(trajB)
        # Probability proportional to e^{-cost}
        cost_logits = -1 * tf.concat([self.costA, self.costB], axis=-1)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=cost_logits, labels=self.label_ph))
        # tf.summary.scalar('loss', self.loss)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.cost_loss = tf.reduce_mean(tf.square(self.costA - self.cost_label_ph))
        # tf.summary.scalar('cost_loss', self.cost_loss)
        self.cost_train_op = tf.train.AdamOptimizer().minimize(self.cost_loss)

        # self.summary_op = tf.merge_all_summaries()
        # self.train_writer = tf.summary.FileWriter()
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if load_path:
            self.load_model(load_path)
        self.num_params = np.sum([np.prod(v.shape) for v in tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)])

    def cost_waypoint(self, waypoint):
        cost = self.sess.run(self.waypoint_cost, feed_dict={
            self.waypoint_ph: waypoint[None],
            K.learning_phase(): False
        })
        return np.squeeze(cost)

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

    def train(self, trajsA, trajsB, labels):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
            self.trajA_ph: trajsA,
            self.trajB_ph: trajsB,
            self.label_ph: labels,
            K.learning_phase(): True
        })
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
        print('Model saved in file: {:s}'.format(save_path))

    def load_model(self, path):
        self.saver.restore(self.sess, path)
