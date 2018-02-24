from mss import mss
import Xlib
import Xlib.display
from PIL import Image
from moviepy.editor import ImageSequenceClip
import numpy as np

import utils
import planners
import constants


def get_geometry():
    '''Should return the position and size of the openrave viewer.
    Position part doesn't really work right now.'''
    display = Xlib.display.Display()
    root = display.screen().root
    # List of all window ids
    wids = root.get_full_property(
        display.intern_atom('_NET_CLIENT_LIST'),
        Xlib.X.AnyPropertyType).value
    # Finding the openrave window by window name
    for wid in wids:
        window = display.create_resource_object('window', wid)
        name = window.get_wm_name()
        if 'OpenRAVE' in name:
            or_window = window
    # Need the parent: https://stackoverflow.com/a/12854004
    parent = or_window.query_tree().parent
    geom = parent.get_geometry()
    x, y = geom.x, geom.y
    width, height = geom.width, geom.height
    return (x, y, width, height)


def record(robot, traj, out_name, fps=60):
    total_duration = traj.GetDuration() # in seconds
    num_frames = fps * (int(total_duration) + 1)

    print('Total dur', total_duration)

    sct = mss()
    monitor = {'top': 25, 'left': 64, 'width': 640, 'height': 480}
    frames = []
    for i in range(num_frames):
        waypoint = traj.Sample((float(i) * total_duration) / num_frames)
        with env:
            robot.SetActiveDOFValues(waypoint[:7])
        sct_img = sct.grab(monitor)
        img = Image.frombytes('RGB', sct_img.size, sct_img.rgb)
        frames.append(np.array(img))
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(out_name, codec='mpeg4', bitrate='1000k')


if __name__ == '__main__':
    env, robot = utils.setup()
    with env:
        wps = planners.trajopt_simple_plan(
            env, robot, constants.configs[1]).GetTraj()
    traj = utils.waypoints_to_traj(env, robot, wps, 1, None)
    record(robot, traj, 'test_video.mp4')
