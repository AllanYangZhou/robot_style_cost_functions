import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras import initializers
from keras import backend as K
import numpy as np

import forward_kinematics
import utils
import constants as c


def jacobian(y_flat, x):
    n = y_flat.shape[0]

    loop_vars = [
        tf.constant(0, tf.int32),
        tf.TensorArray(tf.float32, size=n),
    ]

    _, jacobian = tf.while_loop(
        lambda j, _: j < n,
        lambda j, result: (j+1, result.write(j, tf.gradients(y_flat[j], x))),
        loop_vars)

    return jacobian.stack()


class MLP:
    def __init__(
            self,
            input_dim,
            h_size=64,
            activation='tanh',
            output_dim=None,
            dropout=0.5,
            batchnorm=False):
        if output_dim is None:
            output_dim = input_dim

        self.model = Sequential()

        self.model.add(Dense(
            h_size,
            input_dim=input_dim,
            activation=activation))
        if batchnorm:
            self.model.add(BatchNormalization())
        if dropout is not None:
            self.model.add(Dropout(dropout))

        self.model.add(Dense(h_size, activation=activation))
        if batchnorm:
            self.model.add(BatchNormalization())
        if dropout is not None:
            self.model.add(Dropout(dropout))

        self.model.add(Dense(output_dim))


    def __call__(self, x):
        return self.model(x)


class CostFunction:
    def __init__(self,
                 robot,
                 load_path=None,
                 num_wps=10,
                 num_dofs=7,
                 # Normalize input values to [-1,1]
                 normalize=True,
                 # Whether to provide MLP (x_t, x_t - x_{t-1}) or (x_t, x_{t-1})
                 provide_vel=True,
                 # Whether to provide MLP the C-space values
                 include_configs=False,
                 # Whether to provide MLP all links or just eff
                 use_all_links=True,
                 # Whether to make cf of form ||y||_2^2
                 quadratic=True,
                 # Preference based update learning rate
                 lr=.001,
                 # Number of hidden units
                 h_size=64,
                 activation='relu',
                 dropout=0.5,
                 batchnorm=False,
                 sess=None):
        self.robot = robot
        self.env = robot.GetEnv()
        self.sess = sess or tf.Session()
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
            tf_world_feature_min = tf.constant(c.world_feature_min)
            tf_world_frange = tf.constant(c.world_frange)
            trajA = (2 * (trajA - tf_world_feature_min) / (tf_world_frange)) - 1.
            trajB = (2 * (trajB - tf_world_feature_min) / (tf_world_frange)) - 1.

        if not use_all_links:
            # Only use end effector feature
            trajA = trajA[:,:,15:18]
            trajB = trajB[:,:,15:18]

        if include_configs:
            # Include the original configs
            normalized_trajA = (2 * self.trajA_ph)/(2*np.pi)
            normalized_trajB = (2 * self.trajB_ph)/(2*np.pi)
            trajA = tf.concat([trajA, normalized_trajA], axis=2)
            trajB = tf.concat([trajB, normalized_trajB], axis=2)

        self.label_ph = tf.placeholder(tf.int32, shape=[None], name='label_ph')

        batch_size = tf.shape(trajA)[0]
        input_dim = 2*int(trajA.shape[-1]) + 1 # (x_t, x_{t-1}, t)
        output_dim = input_dim if quadratic else 1
        self.mlp = MLP(
            input_dim,
            h_size=h_size,
            activation=activation,
            output_dim=output_dim,
            dropout=dropout,
            batchnorm=batchnorm)
        self.mlp_outA, self.mlp_outB = [], []
        for i in range(num_wps):
            currA_state = trajA[:,i,:]
            prevA_state = trajA[:,max(i-1,0),:]
            velA = (prevA_state - currA_state) if provide_vel else prevA_state
            currB_state = trajB[:,i,:]
            prevB_state = trajB[:,max(i-1,0),:]
            velB = (prevB_state - currB_state) if provide_vel else prevB_state
            wp_num = tf.fill([batch_size, 1], float(i))
            self.mlp_outA.append(self.mlp(tf.concat([currA_state, velA, wp_num], axis=1)))
            self.mlp_outB.append(self.mlp(tf.concat([currB_state, velB, wp_num], axis=1)))
        self.mlp_outA = tf.concat(self.mlp_outA, axis=1)
        self.mlp_outB = tf.concat(self.mlp_outB, axis=1)
        if quadratic:
            self.costA = tf.reduce_sum(tf.square(self.mlp_outA), axis=1)
            self.costB = tf.reduce_sum(tf.square(self.mlp_outB), axis=1)
        else:
            self.costA = tf.reduce_sum(self.mlp_outA, axis=1)
            self.costB = tf.reduce_sum(self.mlp_outB, axis=1)

        # WARNING: Only the Jacobian for the first trajectory in the batch.
        self.grad_mlp_outA = tf.reshape(
            jacobian(self.mlp_outA[0], self.trajA_ph),
            [input_dim*num_wps, -1])

        # Probability proportional to e^{-cost}
        cost_logits = -1 * tf.stack([self.costA, self.costB], axis=1)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=cost_logits, labels=self.label_ph))
        self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

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


    def get_traj_cost(self, trajs):
        '''Returns cost for a batch of trajs.
        In: (num_trajs, num_wps, 7)
        Out: (num_trajs,)
        '''
        cost = self.sess.run(self.costA, feed_dict={
            self.trajA_ph: trajs,
            K.learning_phase(): False
        })
        return np.squeeze(cost)


    def get_corrcoef(self, test_data, labels):
        '''Correlation between model predictions and labels.'''
        return np.corrcoef(self.get_traj_cost(test_data), labels)[0,1]


    def train_pref(self, trajsA, trajsB, labels):
        '''Train from labeled preferences.'''
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
            self.trajA_ph: trajsA,
            self.trajB_ph: trajsB,
            self.label_ph: labels,
            K.learning_phase(): True
        })
        return loss


    def save_model(self, path, step=None):
        if step:
            self.saver.save(self.sess, path, global_step=step)
        else:
            self.saver.save(self.sess, path)


    def load_model(self, path):
        self.saver.restore(self.sess, path)
