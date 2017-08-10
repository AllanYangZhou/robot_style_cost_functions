import tensorflow as tf
from keras.layers import Dense, Dropout, LeakyReLU
from keras.models import Sequential
from keras import backend as K
import numpy as np


class MLP:
    def __init__(self, input_dim, h_size=64):
        self.model = Sequential()
        self.model.add(Dense(h_size, input_dim=input_dim))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(h_size))
        self.model.add(LeakyReLU())

        self.model.add(Dropout(0.5))
        self.model.add(Dense(1))

    def __call__(self, x):
        return self.model(x)


class CostFunction:
    def __init__(self, load_path=None, num_wps=10, num_dofs=10, per_waypoint=False):
        self.sess = tf.Session()
        self.trajA_ph = tf.placeholder(tf.float32, shape=[None, num_wps, num_dofs], name='trajA_ph')
        self.trajB_ph = tf.placeholder(tf.float32, shape=[None, num_wps, num_dofs], name='trajB_ph')
        self.label_ph = tf.placeholder(tf.int32, shape=[None], name='label_ph')
        self.score_label_ph = tf.placeholder(tf.float32, shape=[None], name='score_label_ph')

        batch_size = tf.shape(self.trajA_ph)[0]
        input_dim = num_dofs if per_waypoint else num_dofs * num_wps
        self.mlp = MLP(input_dim)
        if per_waypoint:
            A_scores, B_scores = [], []
            for wp in range(num_wps):
                A_scores.append(self.mlp(self.trajA_ph[:,wp,:]))
                B_scores.append(self.mlp(self.trajB_ph[:,wp,:]))
            self.scoreA = tf.add_n(A_scores)
            self.scoreB = tf.add_n(B_scores)
        else:
            trajA = tf.reshape(self.trajA_ph, [batch_size, input_dim])
            trajB = tf.reshape(self.trajB_ph, [batch_size, input_dim])
            self.scoreA, self.scoreB = self.mlp(trajA), self.mlp(trajB)
        # Probability proportional to e^{-cost}
        score_logits = -1 * tf.concat([self.scoreA, self.scoreB], axis=-1)
        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=score_logits, labels=self.label_ph))
        # tf.summary.scalar('loss', self.loss)
        self.train_op = tf.train.AdamOptimizer().minimize(self.loss)
        self.score_loss = tf.reduce_mean(tf.square(self.scoreA - self.score_label_ph))
        # tf.summary.scalar('score_loss', self.score_loss)
        self.score_train_op = tf.train.AdamOptimizer().minimize(self.score_loss)

        # self.summary_op = tf.merge_all_summaries()
        # self.train_writer = tf.summary.FileWriter()
        init_op = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess.run(init_op)
        if load_path:
            self.load_model(load_path)
        self.num_params = np.sum([np.prod(v.shape) for v in tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES)])

    def score(self, waypoints):
        # Returns cost for a single trajectory with shape (num_wps, 7)
        score = self.sess.run(self.scoreA, feed_dict={
            self.trajA_ph: waypoints[None],
            K.learning_phase(): False
        })
        return score[0][0]

    def train(self, trajsA, trajsB, labels):
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
            self.trajA_ph: trajsA,
            self.trajB_ph: trajsB,
            self.label_ph: labels,
            K.learning_phase(): True
        })
        return loss

    def pretrain_on_prior_cost(self, trajs, score_labels):
        score_loss, _ = self.sess.run([self.score_loss, self.score_train_op], feed_dict={
            self.trajA_ph: trajs,
            self.score_label_ph: score_labels,
            K.learning_phase(): True
        })
        return score_loss

    def get_score_loss(self, trajs, score_labels):
        score_loss = self.sess.run(self.score_loss, feed_dict={
            self.trajA_ph: trajs,
            self.score_label_ph: score_labels,
            K.learning_phase(): False
        })
        return score_loss

    def save_model(self, path):
        save_path = self.saver.save(self.sess, path)
        print('Model saved in file: {:s}'.format(save_path))

    def load_model(self, path):
        self.saver.restore(self.sess, path)
