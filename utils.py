import interactpy
import openravepy as orpy
import numpy as np
import constants as c


def setup():
    env, robot = interactpy.initialize()
    robot.SetActiveDOFs(robot.arm.GetArmIndices())
    robot.SetActiveDOFValues(c.starting_angles)
    robot.SetDOFValues(c.starting_finger_angles, [7,8,9])
    table = env.GetKinBody('table')
    mug = env.GetKinBody('mug')
    env.Remove(table)
    env.Remove(mug)
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


def waypoints_to_traj(env, robot, waypoints):
    traj = orpy.RaveCreateTrajectory(env, '')
    traj.Init(robot.GetActiveConfigurationSpecification('linear'))
    for i in range(waypoints.shape[0]):
        traj.Insert(i, waypoints[i])
    orpy.planningutils.RetimeActiveDOFTrajectory(
        traj, robot)
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
