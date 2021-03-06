import interactpy
import openravepy as orpy
import numpy as np
import constants as c
import trajoptpy.math_utils as mu
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def setup(objects=False, render=True, two_robots=False):
    env, robot = interactpy.initialize(render=render, two_robots=two_robots)
    robot.SetActiveDOFs(robot.arm.GetArmIndices())
    robot.SetActiveDOFValues(c.configs[0])
    robot.SetDOFValues(c.starting_finger_angles, [7,8,9])
    if two_robots:
        robot2 = env.GetRobots()[1]
        robot2.SetActiveDOFs(robot2.arm.GetArmIndices())
        robot2.SetActiveDOFValues(c.configs[0])
        robot2.SetDOFValues(c.starting_finger_angles, [7,8,9])
    table = env.GetKinBody('table')
    mug = env.GetKinBody('mug')
    env.Remove(table)
    env.Remove(mug)
    if objects:
        env.Load('{:s}/table.xml'.format(c.iact_ctrl_path))
        env.Load('{:s}/cabinet.xml'.format(c.iact_ctrl_path))
        table = env.GetKinBody('table')
        cabinet = env.GetKinBody('cabinet')
        cabinet.SetTransform(c.cabinet_T)
        table.SetTransform(c.table_T)
        cabinet.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(c.cabinet_color)
        table.GetLinks()[0].GetGeometries()[0].SetDiffuseColor(c.table_color)
    env.Load('/usr/local/share/openrave-0.9/data/box1.kinbody.xml')
    box = env.GetKinBody('box1')
    box.SetTransform(c.box_T)

    if two_robots:
        robot_T = robot.GetTransform()
        robot_T[2,-1] -= 1.5
        robot2.SetTransform(robot_T)
        box_T = box.GetTransform()
        box_T[2,-1] -= 1.5
        box2 = clone_kinbody(box)
        box2.SetTransform(box_T)
        robot2.Grab(box2)
    robot.Grab(box)

    if render:
        viewer = env.GetViewer()
        viewer.SetCamera(c.two_robot_camera_T if two_robots else c.camera_T)

    ikmodel3D = (
        orpy.databases.inversekinematics.InverseKinematicsModel(
            robot, iktype=orpy.IkParameterization.Type.Translation3D))
    if not ikmodel3D.load():
        ikmodel3D.autogenerate()
    return (env, robot, robot2) if two_robots else (env, robot)


def get_ik_solns(robot, goal_T):
    with robot.GetEnv():
        ik_param = orpy.IkParameterization(
            goal_T, orpy.IkParameterizationType.Transform6D)
        ik_solutions = robot.arm.FindIKSolutions(
            ik_param, orpy.IkFilterOptions.CheckEnvCollisions,
            ikreturn=False, releasegil=True)
        return ik_solutions


def get_ee_transform(robot, dofs):
    current_dofs = robot.GetActiveDOFValues()
    robot.SetActiveDOFValues(dofs)
    values = robot.arm.hand.GetTransform()
    robot.SetActiveDOFValues(current_dofs)
    return values


def get_link_coords(robot, dofs):
    current_dofs = robot.GetActiveDOFValues()
    robot.SetActiveDOFValues(dofs)
    vals = np.stack([link.GetTransform()[:3,-1]
                     for link in robot.GetLinks()[2:8]])
    robot.SetActiveDOFValues(current_dofs)
    return vals


def get_ee_orientation(robot, dofs):
    current_dofs = robot.GetActiveDOFValues()
    robot.SetActiveDOFValues(dofs)
    ee_dir = -1 * robot.GetJoints()[6].GetAxis()
    robot.SetActiveDOFValues(current_dofs)
    return ee_dir


def get_ee_coords(robot, dofs):
    return get_ee_transform(robot, dofs)[:3,-1]


def constant_timing(waypoints, robot, duration):
    n = waypoints.shape[0]
    dt = float(duration) / (n - 1)
    return np.array([0] + (n - 1) * [dt])


def constant_ee_timing(waypoints, robot, duration):
    ee_positions = np.stack([get_ee_coords(robot, x)
                             for x in waypoints])
    ee_dist = np.linalg.norm(np.diff(ee_positions, axis=0), axis=1)
    total_ee_dist = np.sum(ee_dist)
    dts = []
    for i in range(waypoints.shape[0]):
        dt = 0
        if i:
            dt = (ee_dist[i-1] / total_ee_dist) * duration
        dts.append(dt)
    return np.array(dts)


def quadratic_ee_timing(waypoints, robot, duration):
    ee_positions = np.stack([get_ee_coords(robot, x)
                             for x in waypoints])
    ee_dist = np.linalg.norm(np.diff(ee_positions, axis=0), axis=1)
    cum_ee_dist = np.cumsum(ee_dist)
    total_ee_dist = np.sum(ee_dist)
    b = np.array([0, total_ee_dist])
    T3 = np.power(duration, 3)
    T2 = np.square(duration)
    A = np.array([[T2, duration], [T3/3, T2/2]])
    x = np.linalg.solve(A, b)
    dts = []
    times = []
    for i in range(waypoints.shape[0]):
        dt = 0
        if i:
            roots = np.roots([x[0] / 3, x[1] / 2, 0, -cum_ee_dist[i-1]])
            roots = roots[roots > 0]
            roots = roots[roots < duration]
            time = roots[0]
            if np.iscomplex(time):
                time = time.real
            dt = time
            if i > 1:
                dt = time - times[-1]
            times.append(time)
        dts.append(dt)
    return np.array(dts)


def retime_traj(env, robot, traj_old, f):
    t_prime = f(np.arange(100))
    traj = orpy.RaveCreateTrajectory(env, '')
    spec = robot.GetActiveConfigurationSpecification('linear')
    indices = robot.GetActiveDOFIndices()
    spec.AddDeltaTimeGroup()
    traj.Init(spec)
    for i in range(100):
        wp = traj_old.Sample((t_prime[i] / 99.) * traj_old.GetDuration())
        dt = 0
        if i:
            dt = traj_old.GetDuration() / 99.
        spec.InsertDeltaTime(wp, dt)
        traj.Insert(i, wp)
    return traj


def waypoints_to_traj(env, robot, waypoints, speed, retimer, use_ee_dist=False):
    traj = orpy.RaveCreateTrajectory(env, '')
    spec = robot.GetActiveConfigurationSpecification('linear')
    indices = robot.GetActiveDOFIndices()
    spec.AddDeltaTimeGroup()
    traj.Init(spec)
    if use_ee_dist:
        ee_positions = np.stack([get_ee_coords(robot, x)
                         for x in waypoints])
        distance = np.sum(np.linalg.norm(np.diff(ee_positions,
                                                 axis=0), axis=1))
    else:
        distance = np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1))
    duration = distance / speed
    dts = constant_timing(waypoints, robot, duration)
    for i in range(waypoints.shape[0]):
        wp = np.empty(spec.GetDOF())
        dt = dts[i]
        spec.InsertDeltaTime(wp, dt)
        spec.InsertJointValues(wp, waypoints[i], robot, indices, 0)
        traj.Insert(i, wp)
    if retimer is None:
        retimer = interp1d([0, 19, 80, 99], [0, 9, 90, 99], kind='quadratic')
    return retime_traj(env, robot, traj, retimer)


def make_orientation_cost(robot):
    local_dir = np.array([1, 0, 0])
    arm_joints = [robot.GetJointFromDOFIndex(idx)
                  for idx in robot.arm.GetArmIndices()]
    def f(x):
        robot.SetActiveDOFValues(x)
        return robot.arm.hand.GetTransform()[:3,:3].dot(local_dir) - np.array([0, 0, -1])
    def dfdx(x):
        robot.SetActiveDOFValues(x)
        world_dir = robot.arm.hand.GetTransform()[:3,:3].dot(local_dir)
        return np.array([np.cross(joint.GetAxis(), world_dir)[:3]
                         for joint in arm_joints]).T.copy()
    return f, dfdx


def interpolate_waypoint(start, waypoint, end, num_steps):
    waypoint_step = (num_steps - 1) // 2
    traj = np.empty((num_steps, 7))
    traj[:waypoint_step+1] = mu.linspace2d(start, waypoint, waypoint_step+1)
    traj[waypoint_step:] = mu.linspace2d(waypoint, end, num_steps - waypoint_step)
    return traj


def check_trajs_equal(traj1, traj2):
    return np.allclose(traj1.GetAllWaypoints2D(), traj2.GetAllWaypoints2D())


def plot_ee_speed(traj, robot):
    ee_coords = []
    dt = traj.GetDuration() / (100 - 1)
    for i in range(100):
        wp = traj.Sample(i * dt)[:7]
        ee_coords.append(get_ee_coords(robot, wp))
    ee_coords = np.stack(ee_coords)
    ee_speed = np.linalg.norm(np.diff(ee_coords, axis=0), axis=1)
    plt.ylim([0, 1.1 * np.max(ee_speed)])
    plt.plot(ee_speed)


def evaluate_costs(robot, waypoints, custom_costs):
    current_dofs = robot.GetActiveDOFValues()
    costs = [sum([cost_fn(wp) for cost_fn in custom_costs.values()])
             for wp in waypoints]
    robot.SetActiveDOFValues(current_dofs)
    return sum(costs)


def append_ee_position(robot, waypoints):
    current_dofs = robot.GetActiveDOFValues()
    ee_positions = []
    base_pos = robot.GetTransform()[:3,-1]
    for wp in waypoints:
        robot.SetActiveDOFValues(wp)
        ee_pos = robot.arm.GetEndEffectorTransform()[:3,-1] - base_pos
        ee_or = robot.GetJoints()[6].GetAxis()
        ee_positions.append(np.concatenate([ee_pos, ee_or]))
    ee_positions = np.stack(ee_positions)
    robot.SetActiveDOFValues(current_dofs)
    return np.concatenate([waypoints, ee_positions], axis=1)


def world_space_featurizer(robot, waypoints):
    link_pos = np.stack([
        get_link_coords(robot, wp).reshape(1, -1).squeeze()
        for wp in waypoints])
    ee_or = np.stack([get_ee_orientation(robot, wp) for wp in waypoints])
    wf = np.concatenate([link_pos, ee_or], axis=1)
    return wf


def normalize_vec(x):
    return x / np.linalg.norm(x)


def get_pos_ik_soln(robot, x):
    ik_param = orpy.IkParameterization(
        x, orpy.IkParameterizationType.Translation3D)
    ik_solution = robot.arm.FindIKSolution(
        ik_param, orpy.IkFilterOptions.CheckEnvCollisions,
        ikreturn=False, releasegil=True)
    return ik_solution


def get_pos_ik_solns(robot, x):
    ik_param = orpy.IkParameterization(
        x, orpy.IkParameterizationType.Translation3D)
    ik_solution = robot.arm.FindIKSolutions(
        ik_param, orpy.IkFilterOptions.CheckEnvCollisions,
        ikreturn=False, releasegil=True)
    return ik_solution


def plot_waypoints(env, robot, waypoints, size=12, color='#ff3300'):
    color_arr = np.array([int(color[i+1:i+3], 16) for i in (0, 2 ,4)]) / 255.
    ee_coords = np.stack([get_ee_coords(robot, wp) for wp in waypoints])
    return env.plot3(ee_coords, size, color_arr)


def num_hess(f, x, eps=1e-5):
    xp = x.copy()
    hess = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        xp[i] = x[i] + (eps/2)
        yp = f(xp)
        xp[i] = x[i] - (eps/2)
        ym = f(xp)
        hess[i] = (yp + ym - 2*y) / ((eps * eps) / 4)
        xp[i] = x[i]
    return hess


def num_diff(f, x, eps=1e-5):
    xp = x.copy()
    grad = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        xp[i] = x[i] + (eps/2.)
        yp = f(xp)
        xp[i] = x[i] - (eps/2.)
        ym = f(xp)
        grad[i] = (yp - ym) / eps
        xp[i] = x[i]
    return grad


class TrainingQueue:
    def __init__(self, maxsize=None):
        self.q = []
        self.maxsize = maxsize


    def sample(self, num=1):
        if num == 1:
            idx = np.random.choice(len(self.q))
            return self.q[idx]
        else:
            idcs = np.random.choice(len(self.q), size=num, replace=False)
            return [self.q[idx] for idx in idcs]


    def add(self, elem):
        self.q.append(elem)
        if self.maxsize and len(self.q) > self.maxsize:
            self.q.pop(0)


    def extend(self, arr):
        for elem in arr:
            self.add(elem)


    def pop(self, idx=0):
        return self.q.pop(idx)


    def __len__(self):
        return len(self.q)


def synthetic_label_func(synthetic_cost):
    def label_func(xA, timeA, xB, timeB):
        desired_time = (2 * (15 - 1) / 29.) - 1.
        cA = synthetic_cost(xA) + (timeA - desired_time) ** 2
        cB = synthetic_cost(xB) + (timeB - desired_time) ** 2
        label = cB < cA
        if np.abs(cA-cB) < .1:
            label = None
        return label
    return label_func


def get_labels(cf,
               data_q,
               training_q,
               label_func,
               num_samples=100,
               time_ratio=0.2,
               adaptive_timing=True):
    i = 0
    while i < num_samples:
        if np.random.uniform(low=0, high=1) < (1. - time_ratio):
            xs = data_q.sample(num=2)
            xA, xB = xs[0], xs[1]
            timeA = timeB = 1
            data = ((xA, timeA), (xB, timeB), label_func(xA, timeA, xB, timeB))
            if data[2] is not None:
                training_q.add(data)
        else:
            xA = data_q.sample(num=1)
            opt_time = (2 * (5 - 1) / 29.) - 1.
            for j in range(1, 31):
                timeB = (2 * (j - 1) / 29.) - 1
                if not np.allclose(opt_time, timeB):
                    data = ((xA, opt_time), (xA, timeB), 0)
                    training_q.add(data)
        i += 1


def train(cf, tq, epochs=5):
    loss = []
    for i in range(epochs * len(tq)):
        xA, xB, label = tq.sample()
        l = cf.train_pref(xA[None], xB[None], [int(label)])
        loss.append(l)
    return np.array(loss)


def add_samples_to_label(robot, to_label, data_q, num=30):
    env = robot.GetEnv()
    for i in range(num):
        xs = data_q.sample(num=2)
        xA, xB = xs[0], xs[1]
        with env:
            trajA = waypoints_to_traj(env, robot, xA[:,:7], 0.5, None)
            trajB = waypoints_to_traj(env, robot, xB[:,:7], 0.5, None)
        to_label.append((xA, trajA, xB, trajB))


def clone_kinbody(kinbody):
    env = kinbody.GetEnv()
    newbody = orpy.RaveCreateKinBody(env, kinbody.GetXMLId())
    newbody.Clone(kinbody, 0)
    env.AddKinBody(newbody, True)
    return newbody


def vel_cost(traj):
    return np.sum(np.square(np.diff(traj, axis=0)))


def random_init_maker(given_init, one_wp=False):
    start = given_init[0]
    end = given_init[-1]
    if one_wp:
        changed_idx = np.random.choice(range(3,7))
        original = given_init[changed_idx]
        new_pt = np.random.multivariate_normal(original, .05*np.eye(7))
        modified_init = np.concatenate([
            mu.linspace2d(given_init[0], new_pt, changed_idx + 1),
            mu.linspace2d(new_pt, given_init[-1], 10 - changed_idx)[1:]
        ], axis=0)
    else:
        modified_init = given_init.copy()
        modified_init[1:-1] = np.random.multivariate_normal(
            given_init[1:-1].reshape(-1),
            np.linalg.norm(start - end) * 0.0025 * np.eye(56)).reshape(8,-1)
    return modified_init


A = np.zeros((56,56))
for i in range(8):
    A[i*7:(i+1)*7, i*7:(i+1)*7] = 2 * np.eye(7)
    if i < 7:
        A[i*7:(i+1)*7, (i+1)*7:(i+2)*7] = -1 * np.eye(7)
        A[(i+1)*7:(i+2)*7, i*7:(i+1)*7] = -1 * np.eye(7)
Ainv = np.linalg.inv(A)

def smooth_perturb(var=.05):
    changed_idx = np.random.choice(range(2,6))
    delta_traj = np.zeros((8, 7))
    new_pt = np.random.multivariate_normal(np.zeros(7), var*np.eye(7))
    delta_traj[changed_idx] = new_pt
    delta_prime = Ainv.dot(delta_traj.reshape(56)).reshape(8, 7)
    const = np.linalg.norm(delta_traj[changed_idx]) / np.linalg.norm(delta_prime[changed_idx])
    delta_prime = const * delta_prime
    delta = np.zeros((10, 7))
    delta[1:-1] = delta_prime
    return delta


def ee_traj_cost(x, robot):
    x = x.reshape((10,7))
    ee_positions = np.stack([get_ee_coords(robot, wp)
                             for wp in x])
    ee_cost = np.sum(np.square(np.diff(ee_positions, axis=0)))
    return ee_cost
