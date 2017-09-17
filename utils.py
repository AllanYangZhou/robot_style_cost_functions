import interactpy
import openravepy as orpy
import numpy as np
import constants as c
import trajoptpy.math_utils as mu
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def setup(objects=False, render=True):
    env, robot = interactpy.initialize(render=render)
    robot.SetActiveDOFs(robot.arm.GetArmIndices())
    robot.SetActiveDOFValues(c.configs[0])
    robot.SetDOFValues(c.starting_finger_angles, [7,8,9])
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
    robot.Grab(box)
    if render:
        viewer = env.GetViewer()
        viewer.SetCamera(c.camera_T)
    ikmodel3D = (
        orpy.databases.inversekinematics.InverseKinematicsModel(
            robot, iktype=orpy.IkParameterization.Type.Translation3D))
    if not ikmodel3D.load():
        ikmodel3D.autogenerate()
    return env, robot


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
                     for link in robot.GetLinks()[:8]])
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


def waypoints_to_traj(env, robot, waypoints, duration, retimer):
    traj = orpy.RaveCreateTrajectory(env, '')
    spec = robot.GetActiveConfigurationSpecification('linear')
    indices = robot.GetActiveDOFIndices()
    spec.AddDeltaTimeGroup()
    traj.Init(spec)
    dts = constant_ee_timing(waypoints, robot, duration)
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
    return np.concatenate([link_pos, ee_or], axis=1)


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
    def __init__(self, maxsize=200):
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
        if len(self.q) > self.maxsize:
            self.q.pop(0)


    def extend(self, arr):
        for elem in arr:
            self.add(elem)


    def __len__(self):
        return len(self.q)


def synthetic_label_func(synthetic_cost):
    def label_func(xA, xB):
        cA = synthetic_cost(xA)
        cB = synthetic_cost(xB)
        label = cB < cA
        if np.abs(cA-cB) < .1:
            label = None
        return label
    return label_func


def get_labels(cf, data_q, training_q, label_func, num_samples=100, time_ratio=0.2):
    i = 0
    while i < num_samples:
        if np.random.uniform(low=0, high=1) < (1. - time_ratio):
            xs = data_q.sample(num=2)
            xA, xB = xs[0], xs[1]
        else:
            xs = data_q.sample(num=1)
            xA = xs.copy()
            xB = xs.copy()
            mean = cf.naive_time_opt(xs[:,:7])
            diff = np.random.normal(0, 1)
            xA[:,7] = ((mean + diff) / 10.) * np.ones(10)
            xB[:,7] = ((mean - diff) / 10.) * np.ones(10)
        data = (xA, xB, label_func(xA, xB))
        if data[2] is not None:
            training_q.add(data)
            i += 1
