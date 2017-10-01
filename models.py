import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.wrappers import TimeDistributed
from keras.models import Sequential
from keras import backend as K
import numpy as np

import forward_kinematics
import utils
import constants as c


class MLP:
    def __init__(self, input_dim, activation='tanh'):
        self.model = Sequential()
        self.model.add(Dense(32, input_dim=input_dim, activation=activation))

        self.model.add(Dropout(0.5))
        self.model.add(Dense(16, activation=activation))

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))


    def __call__(self, x):
        return self.model(x)


class CostFunction:
    def __init__(self,
                 robot,
                 load_path=None,
                 num_wps=10,
                 num_dofs=7,
                 normalize=False,
                 activation='tanh'):
        self.robot = robot
        self.env = robot.GetEnv()
        self.sess = tf.Session()
        self.trajA_ph = tf.placeholder(tf.float32,
                                       shape=[None, num_wps, num_dofs],
                                       name='trajA_ph')
        self.trajB_ph = tf.placeholder(tf.float32,
                                       shape=[None, num_wps, num_dofs],
                                       name='trajB_ph')
        g0s, axes, anchors = forward_kinematics.openrave_get_fk_params(robot)
        trajA = forward_kinematics.augment_traj(self.trajA_ph, g0s, axes, anchors)
        trajB = forward_kinematics.augment_traj(self.trajB_ph, g0s, axes, anchors)
        if normalize:
            raise Exception('Not ready yet!')
            tf_world_feature_min = tf.constant(c.world_feature_min.astype(np.float32))
            tf_world_frange = tf.constant(c.world_frange.astype(np.float32))
            trajA = (2 * (trajA - tf_world_feature_min) / (tf_world_frange)) - 1.
            trajB = (2 * (trajB - tf_world_feature_min) / (tf_world_frange)) - 1.
        self.label_ph = tf.placeholder(tf.int32, shape=[None], name='label_ph')
        self.cost_label_ph = tf.placeholder(tf.float32, shape=[None], name='cost_label_ph')
        self.gan_label_ph = tf.placeholder(tf.float32, shape=[None], name='gan_label_ph')

        batch_size = tf.shape(trajA)[0]
        self.mlp =  MLP(2*int(trajA.shape[-1]) + 1, activation=activation)
        self.mlp_outA, self.mlp_outB = [], []
        for i in range(num_wps):
            prev_idx = max(i-1, 0)
            currA_state = trajA[:,i,:]
            velA = trajA[:,prev_idx,:] - currA_state
            currB_state = trajB[:,i,:]
            velB = trajB[:,prev_idx,:] - currB_state
            wp_num = tf.fill([batch_size, 1], float(i))
            self.mlp_outA.append(self.mlp(tf.concat([currA_state, velA, wp_num], axis=1)))
            self.mlp_outB.append(self.mlp(tf.concat([currB_state, velB, wp_num], axis=1)))
        self.mlp_outA = tf.stack(self.mlp_outA, axis=1)
        self.mlp_outB = tf.stack(self.mlp_outB, axis=1)
        self.costA = tf.reduce_sum(tf.square(self.mlp_outA), axis=1)
        self.costB = tf.reduce_sum(tf.square(self.mlp_outB), axis=1)

        #self.grad_mlp_outA = tf.gradients(self.mlp_outA[:,0,0], self.trajA_ph)
        grads = []
        for i in range(num_wps):
            grad_i = tf.gradients(self.mlp_outA[:,i,0], self.trajA_ph)[0]
            grads.append(tf.reshape(grad_i, [num_wps * num_dofs]))
        self.grad_mlp_outA = tf.stack(grads)
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


    def get_mlp_out(self, waypoints):
        mlp_out = self.sess.run(self.mlp_outA, feed_dict={
            self.trajA_ph: waypoints[None],
            K.learning_phase(): False
        })
        return np.squeeze(mlp_out)


    def get_grad_mlp_out(self, waypoints):
        grad = self.sess.run(self.grad_mlp_outA, feed_dict={
            self.trajA_ph: waypoints[None],
            K.learning_phase(): False
        })
        return np.squeeze(grad)


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


    def train_cost(self, trajs, cost_labels):
        cost_loss, _ = self.sess.run([self.cost_loss, self.cost_train_op], feed_dict={
            self.trajA_ph: trajs,
            self.cost_label_ph: cost_labels,
            K.learning_phase(): True
        })
        return cost_loss


    def train_gan_cost(self, traj, gan_labels):
        return self.sess.run([self.gan_cost_loss], feed_dict={
            self.trajA_ph: traj,
            self.gan_label_ph: gan_labels,
            K.learning_phase(): True
        })


    def get_cost_loss(self, trajs, cost_labels):
        cost_loss = self.sess.run(self.cost_loss, feed_dict={
            self.trajA_ph: traj,
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
