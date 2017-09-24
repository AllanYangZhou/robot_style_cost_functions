import tensorflow as tf
import numpy as np


def openrave_get_fk_params(robot):
    g0s = []
    axes, anchors = [], []
    with robot.GetEnv():
        current = robot.GetActiveDOFValues()
        robot.SetActiveDOFValues(np.zeros(7))
        for link in robot.GetLinks()[1:8]:
            g0s.append(link.GetTransform())
        for j in robot.GetJoints()[:7]:
            axes.append(j.GetAxis())
            anchors.append(j.GetAnchor())
        robot.SetActiveDOFValues(current)
    params = {
        'g0': g0s,
        'axis': axes,
        'anchor': anchors
    }
    return params


def rodrigues(w_hat, theta):
    return tf.eye(3) + tf.sin(theta) * w_hat + tf.matmul(w_hat, w_hat) * (1-tf.cos(theta))


def make_homog(rot, trans):
    top = tf.concat([rot, trans], axis=1)
    return tf.concat([top, tf.convert_to_tensor([[0., 0, 0, 1]])], axis=0)


def top_right(rot, w, v, theta):
    term1 = tf.matmul(tf.eye(3) - rot, tf.transpose(tf.cross(w, v)[None]))
    wwT = tf.matmul(tf.transpose(w[None]), w[None])
    term2 = theta * tf.matmul(wwT, tf.transpose(v[None]))
    return term1 + term2


def forward_kinematics(configs, params):
    num_links = len(params['axis'])
    results = []
    g_1 = tf.constant(np.eye(4).astype(np.float32))
    for i in range(num_links):
        g0 = tf.constant(params['g0'][i].astype(np.float32))
        w = tf.constant(params['axis'][i].astype(np.float32))
        q = tf.constant(params['anchor'][i].astype(np.float32))
        v = -1 * tf.cross(w, q)
        w_hat = tf.convert_to_tensor([[0, -w[2], w[1]],
                                      [w[2], 0, -w[0]],
                                      [-w[1], w[0], 0]])
        rot = rodrigues(w_hat, configs[i])
        tr = top_right(rot, w, v, configs[i])
        homog = make_homog(rot, tr)
        g_1 = tf.matmul(g_1, homog)
        res = tf.matmul(g_1, g0)
        results.append(res)
    return results
