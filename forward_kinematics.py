import tensorflow as tf
import numpy as np


def openrave_get_fk_params(robot):
    g0s = []
    axes, anchors = [], []
    with robot.GetEnv():
        current = robot.GetActiveDOFValues()
        robot.SetActiveDOFValues(np.zeros(7))
        for link in robot.GetLinks()[1:8]:
            g0s.append(tf.constant(link.GetTransform().astype(np.float32)))
        for j in robot.GetJoints()[:7]:
            axes.append(tf.constant(j.GetAxis().astype(np.float32)))
            anchors.append(tf.constant(j.GetAnchor().astype(np.float32)))
        robot.SetActiveDOFValues(current)
    return g0s, axes, anchors


def rodrigues(w_hat, theta):
    t1 = tf.eye(3)
    t2 = tf.sin(theta)[:,None,None] * w_hat[None]
    t3 = (1-tf.cos(theta))[:,None,None] * tf.matmul(w_hat, w_hat)[None]
    return t1 + t2 + t3


def make_homog(rot, trans):
    top = tf.concat([rot, trans], axis=2)
    batch_size = tf.shape(rot)[0]
    bottom = tf.convert_to_tensor([[0., 0, 0, 1]])
    bottom = tf.tile(bottom[None], [batch_size, 1, 1])
    return tf.concat([top, bottom], axis=1)


def top_right(rot, w, v, theta):
    batch_size = tf.shape(rot)[0]
    w_cross_v = tf.tile(tf.transpose(tf.cross(w, v)[None])[None], [batch_size, 1, 1])
    term1 = tf.matmul(tf.eye(3) - rot, w_cross_v)
    wwT = tf.matmul(tf.transpose(w[None]), w[None])
    term2 = theta[:,None,None] * tf.matmul(wwT, tf.transpose(v[None]))[None]
    return term1 + term2


def forward_kinematics(configs, g0s, axes, anchors):
    batch_size = tf.shape(configs)[0]
    num_links = len(axes)
    results = []
    g_1 = tf.tile(tf.eye(4)[None], [batch_size, 1, 1])
    I3 = tf.eye(3)
    for i in range(num_links):
        g0 = tf.tile(g0s[i][None], [batch_size, 1, 1])
        w = axes[i]
        q = anchors[i]
        v = -1 * tf.cross(w, q)
        tiled_w = tf.reshape(tf.tile(w, [3]), [3, 3])
        w_hat = tf.transpose(tf.cross(tiled_w, I3))
        rot = rodrigues(w_hat, configs[:,i])
        tr = top_right(rot, w, v, configs[:,i])
        homog = make_homog(rot, tr)
        g_1 = tf.matmul(g_1, homog)
        res = tf.matmul(g_1, g0)
        results.append(res)
    return results


def augment_traj(traj, g0s, axes, anchors):
    num_wps = int(traj.shape[1])
    num_dofs = int(traj.shape[2])
    reshaped = tf.reshape(traj, [-1, num_dofs])
    results = forward_kinematics(reshaped, g0s, axes, anchors)
    augments = []
    for res in results[1:]:
        pos = res[:,:3,-1]
        pos_reshaped = tf.reshape(pos, [-1, num_wps, 3])
        augments.append(pos_reshaped)
    augments.append(augments[-2] - augments[-1])
    augments = tf.concat(augments, axis=-1)
    return augments
