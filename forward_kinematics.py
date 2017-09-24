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
    return tf.eye(3) + tf.sin(theta) * w_hat + tf.matmul(w_hat, w_hat) * (1-tf.cos(theta))


def make_homog(rot, trans):
    top = tf.concat([rot, trans], axis=1)
    return tf.concat([top, tf.convert_to_tensor([[0., 0, 0, 1]])], axis=0)


def top_right(rot, w, v, theta):
    term1 = tf.matmul(tf.eye(3) - rot, tf.transpose(tf.cross(w, v)[None]))
    wwT = tf.matmul(tf.transpose(w[None]), w[None])
    term2 = theta * tf.matmul(wwT, tf.transpose(v[None]))
    return term1 + term2


def forward_kinematics(configs, g0s, axes, anchors):
    num_links = len(axes)
    results = []
    g_1 = tf.constant(np.eye(4).astype(np.float32))
    for i in range(num_links):
        g0 = g0s[i]
        w = axes[i]
        q = anchors[i]
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


def forward_kinematics_traj(traj, num_wps, g0s, axes, anchors):
    wfs = []
    for i in range(num_wps):
        res = forward_kinematics(traj[i,:], g0s, axes, anchors)
        wf = tf.stack([r[:3,-1] for r in res[1:]])
        wf = tf.reshape(wf, [-1])
        wfs.append(wf)
    return tf.stack(wfs)
