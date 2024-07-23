import numpy as np
import matplotlib.pyplot as plt
from utils import DLT
# plt.style.use('seaborn')
plt.style.use("seaborn-v0_8")

pose_keypoints = np.array([16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28])

def read_keypoints(filename):
    with open(filename, 'r') as fin:
        kpts = []
        while True:
            line = fin.readline()
            if line == '': 
                break

            line = line.split()
            line = [float(s) for s in line]
            line = np.reshape(line, (len(pose_keypoints), -1))
            kpts.append(line)

        kpts = np.array(kpts)
    return kpts

paused = False
current_frame = 0

def on_key(event):
    global paused, current_frame, p3ds
    if event.key == ' ':
        paused = not paused
    elif event.key == 'left' and paused:
        current_frame = max(0, current_frame - 1)
        draw_frame(p3ds, current_frame)
    elif event.key == 'right' and paused:
        current_frame = min(len(p3ds) - 1, current_frame + 1)
        draw_frame(p3ds, current_frame)

def draw_frame(p3ds, frame_index):
    ax.cla()
    torso = [[0, 1], [1, 7], [7, 6], [6, 0]]
    armr = [[1, 3], [3, 5]]
    arml = [[0, 2], [2, 4]]
    legr = [[6, 8], [8, 10]]
    legl = [[7, 9], [9, 11]]
    body = [torso, arml, armr, legr, legl]
    colors = ['red', 'blue', 'green', 'black', 'orange']

    kpts3d = p3ds[frame_index]
    for bodypart, part_color in zip(body, colors):
        for _c in bodypart:
            ax.plot(xs=[kpts3d[_c[0], 0], kpts3d[_c[1], 0]], ys=[kpts3d[_c[0], 1], kpts3d[_c[1], 1]], zs=[kpts3d[_c[0], 2], kpts3d[_c[1], 2]], linewidth=4, c=part_color)

    # Uncomment these if you want scatter plot of keypoints and their indices.
    for i in range(12):
        ax.text(kpts3d[i, 0], kpts3d[i, 1], kpts3d[i, 2], str(i))
        ax.scatter(xs=kpts3d[i:i+1, 0], ys=kpts3d[i:i+1, 1], zs=kpts3d[i:i+1, 2])

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    ax.set_xlim3d(-10, 10)
    ax.set_xlabel('x')
    ax.set_ylim3d(-10, 10)
    ax.set_ylabel('y')
    ax.set_zlim3d(-10, 10)
    ax.set_zlabel('z')

    # Add frame counter
    ax.text2D(0.05, 0.95, f"{frame_index + 1}/{len(p3ds)}", transform=ax.transAxes)

    plt.draw()

def visualize_3d(p3ds):
    """Now visualize in 3D"""
    global ax
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    fig.canvas.mpl_connect('key_press_event', on_key)

    for framenum in range(len(p3ds)):
        if framenum % 2 == 0:
            continue  # skip every 2nd frame
        while paused:
            plt.pause(0.1)
        draw_frame(p3ds, framenum)
        current_frame = framenum
        plt.pause(0.1)
        
if __name__ == '__main__':
    p3ds = read_keypoints('kpts_3d.dat')
    visualize_3d(p3ds)