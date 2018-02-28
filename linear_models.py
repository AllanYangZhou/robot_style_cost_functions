import tensorflow as tf
from keras.layers import Dense
from keras.models import Sequential
from keras import initializers
import numpy as np


def featurizer(robot, traj):
    current = robot.GetActiveDOFValues()
    env = robot.GetEnv()
    with env:
        features = []
        for i in range(10):
            feature_vals = np.empty(4)
            x = traj[i]
            robot.SetActiveDOFValues(x)
            feature_vals[0] = np.linalg.norm(robot.arm.hand.GetTransform()[:2,3])
            feature_vals[1] = (np.square(robot.arm.hand.GetTransform()[2,3] + .55)-1)
            feature_vals[2] = np.dot(robot.GetJoints()[6].GetAxis(), np.array([0, 0, 1]))
            if i:
                feature_vals[3] = np.linalg.norm(feature_vals[:3] - features[-1][:3])
            else:
                feature_vals[3] = 0
            features.append(feature_vals)
        robot.SetActiveDOFValues(current)
    return np.stack(features)


class Linear:
    def __init__(self, input_dim):
        # self.W = tf.Variable(np.zeros(input_dim))
        self.model = Sequential()
        self.model.add(Dense(1, input_dim=input_dim, use_bias=False))


    def __call__(self, x):
        # return self.W * x
        return self.model(x)


class LinearCostFunction:
    def __init__(self,
                 robot,
                 load_path=None,
                 num_wps=10,
                 num_dofs=4,
                 per_waypoint=False):
        self.robot = robot
        self.sess = tf.Session()
        self.trajA_ph = tf.placeholder(tf.float32,
                                       shape=[None, num_wps, num_dofs],
                                       name='trajA_ph')
        self.trajB_ph = tf.placeholder(tf.float32,
                                       shape=[None, num_wps, num_dofs],
                                       name='trajB_ph')
        self.label_ph = tf.placeholder(tf.int32, shape=[None], name='label_ph')

        trajA = self.trajA_ph
        trajB = self.trajB_ph
        if not per_waypoint:
            trajA = tf.concat([
                tf.reduce_sum(trajA[:,:,:3], axis=1),
                trajA[:,:,3]], axis=1)
            trajB = tf.concat([
                tf.reduce_sum(trajB[:,:,:3], axis=1),
                trajB[:,:,3]], axis=1)
            # trajA = tf.reshape(trajA, [-1, num_wps * num_dofs])
            # trajB = tf.reshape(trajB, [-1, num_wps * num_dofs])

            # self.mlp = Linear(num_wps * num_dofs)
            self.mlp = Linear(3 + num_wps)
            self.costA = self.mlp(trajA)
            self.costB = self.mlp(trajB)
            cost_logits = -1 * tf.concat([self.costA, self.costB], axis=1)
        else:
            self.mlp = Linear(num_dofs)
            self.mlp_outA, self.mlp_outB = [], []
            for i in range(num_wps):
                currA_state = trajA[:,i,:]
                currB_state = trajB[:,i,:]
                self.mlp_outA.append(self.mlp(currA_state))
                self.mlp_outB.append(self.mlp(currB_state))
            self.mlp_outA = tf.concat(self.mlp_outA, axis=1)
            self.mlp_outB = tf.concat(self.mlp_outB, axis=1)
            self.costA = tf.reduce_sum(self.mlp_outA, axis=1)
            self.costB = tf.reduce_sum(self.mlp_outB, axis=1)
            cost_logits = -1 * tf.stack([self.costA, self.costB], axis=1)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=cost_logits, labels=self.label_ph))
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)

        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables())
        self.sess.run(init_op)
        if load_path:
            self.load_model(load_path)
        self.num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])


    def _featurize_trajs(self, trajs):
        featurized_trajs = []
        for traj in trajs:
            featurized_trajs.append(
                featurizer(self.robot, traj))
        return np.stack(featurized_trajs)


    def get_traj_cost(self, trajs):
        '''Returns cost for a batch of trajs.
        In: (num_trajs, num_wps, 7)
        Out: (num_trajs,)
        '''
        cost = self.sess.run(self.costA, feed_dict={
            self.trajA_ph: self._featurize_trajs(trajs)
        })
        return np.squeeze(cost)


    def get_corrcoef(self, test_data, labels):
        '''Correlation between model predictions and labels.'''
        return np.corrcoef(self.get_traj_cost(test_data), labels)[0,1]


    def train_pref(self, trajsA, trajsB, labels):
        '''Train from labeled preferences.'''
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
            self.trajA_ph: self._featurize_trajs(trajsA),
            self.trajB_ph: self._featurize_trajs(trajsB),
            self.label_ph: labels
        })
        return loss


    def save_model(self, path, step=None):
        if step:
            self.saver.save(self.sess, path, global_step=step)
        else:
            self.saver.save(self.sess, path)


    def load_model(self, path):
        self.saver.restore(self.sess, path)
