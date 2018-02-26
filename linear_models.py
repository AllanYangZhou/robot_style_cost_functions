import tensorflow as tf
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras import initializers
from keras import backend as K
import numpy as np


def featurizer(robot, traj):
    current = robot.GetActiveDOFValues()
    env = robot.GetEnv()
    with env:
        features = []
        for i in range(10):
            feature_vals = np.empty(5)
            x = traj[i]
            robot.SetActiveDOFValues(x)
            feature_vals[:3] = robot.arm.hand.GetTransform()[:3,3]
            feature_vals[3] = np.dot(robot.GetJoints()[6].GetAxis(), np.array([0, 0, 1]))
            if i:
                feature_vals[4] = np.linalg.norm(feature_vals[:3] - features[-1][:3])
            else:
                feature_vals[4] = 0
            features.append(feature_vals)
        robot.SetActiveDOFValues(current)
    return np.concatenate(features)


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
                 num_dofs=50):
        self.robot = robot
        self.sess = tf.Session()
        self.trajA_ph = tf.placeholder(tf.float32,
                                       shape=[None, num_dofs],
                                       name='trajA_ph')
        self.trajB_ph = tf.placeholder(tf.float32,
                                       shape=[None, num_dofs],
                                       name='trajB_ph')
        self.label_ph = tf.placeholder(tf.int32, shape=[None], name='label_ph')

        self.mlp = Linear(num_dofs)
        self.costA = self.mlp(self.trajA_ph)
        self.costB = self.mlp(self.trajB_ph)

        # Probability proportional to e^{-cost}
        cost_logits = -1 * tf.concat([self.costA, self.costB], axis=1)
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
            self.trajA_ph: self._featurize_trajs(trajs),
            K.learning_phase(): False
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
