import interactpy
import openravepy as orpy
import numpy as np
import constants as c
import trajoptpy.math_utils as mu

def setup(objects=False):
    env, robot = interactpy.initialize()
    robot.SetActiveDOFs(robot.arm.GetArmIndices())
    robot.SetActiveDOFValues(c.starting_angles)
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
    viewer = env.GetViewer()
    viewer.SetCamera(c.camera_T)
    return env, robot


def get_ik_solns(robot, goal_T):
    with robot.GetEnv():
        ik_param = orpy.IkParameterization(
            goal_T, orpy.IkParameterizationType.Transform6D)
        ik_solutions = robot.arm.FindIKSolutions(
            ik_param, orpy.IkFilterOptions.CheckEnvCollisions,
            ikreturn=False, releasegil=True)
        return ik_solutions


def get_ee_coords(robot, dofs):
    current_dofs = robot.GetActiveDOFValues()
    robot.SetActiveDOFValues(dofs)
    values = robot.arm.hand.GetTransform()[:3,-1]
    robot.SetActiveDOFValues(current_dofs)
    return values


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


def waypoints_to_traj(env, robot, waypoints, duration):
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
    return traj


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
