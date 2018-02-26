from mss import mss
import Xlib
import Xlib.display
from PIL import Image
import numpy as np
import imageio

import utils
import planners
import constants


def get_geometry():
    '''Should return the position and size of the openrave viewer.'''
    display = Xlib.display.Display()
    root = display.screen().root
    # List of all window ids
    wids = root.get_full_property(
        display.intern_atom('_NET_CLIENT_LIST'),
        Xlib.X.AnyPropertyType).value
    # Finding the openrave window by window name
    for wid in wids:
        window = display.create_resource_object('window', wid)
        # NOTE: if any of the windows have non-ascii chars in the
        # name, e.g. certain web page titles, then this will crash.
        name = window.get_wm_name()
        if 'OpenRAVE' in name:
            or_window = window
    # Need the parent: https://stackoverflow.com/a/12854004
    parent = or_window.query_tree().parent
    geom = window.get_geometry()
    x, y = geom.x, geom.y
    translated_data = window.translate_coords(root, x, y)
    x = -1 * translated_data.x
    y = -1 * translated_data.y
    width, height = geom.width, geom.height
    return {'top': y, 'left': x, 'width': width, 'height': height}


def record(robot, traj, out_name, fps=60, monitor=None):
    current_config = robot.GetActiveDOFValues()
    env = robot.GetEnv()
    total_duration = traj.GetDuration() # in seconds
    num_frames = fps * (int(total_duration) + 1)

    frames = []
    with mss() as sct:
        if monitor is None:
            monitor = {'top': 25, 'left': 64, 'width': 640, 'height': 480}
        for i in range(num_frames):
            waypoint = traj.Sample((float(i) * total_duration) / num_frames)
            with env:
                robot.SetActiveDOFValues(waypoint[:7])
            sct_img = sct.grab(monitor)
            img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
            frames.append(np.array(img))
    with env:
        robot.SetActiveDOFValues(current_config)
    imageio.mimsave(out_name, frames, fps=60)


if __name__ == '__main__':
    env, robot = utils.setup()
    monitor = get_geometry()
    with env:
        wps = planners.trajopt_simple_plan(
            env, robot, constants.configs[1]).GetTraj()
    traj = utils.waypoints_to_traj(env, robot, wps, 1, None)
    record(robot, traj, 'test_video.mp4', monitor=monitor)
