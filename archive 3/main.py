import numpy as np
import random
from vispy import app, scene
from vispy.geometry import Rect
from funcs import init_boids, directions, propagate, flocking
app.use_app('PyQt5')

w, h = 1280, 960
N = 100
dt = 0.1
asp = w / h
perception = 1/20
# walls_order = 8
better_walls_w = 0.05
vrange=(0, 0.1)
arange=(0, 0.05)

cnt_in_a_view = 10
cnt_rely_on = 5

#                    c      a    s      w
coeffs = np.array([0.05, 0.02,   0.1,  0.03])

# 0  1   2   3   4   5
# x, y, vx, vy, ax, ay
boids = np.zeros((N, 6), dtype=np.float64)
init_boids(boids, asp, vrange=vrange)
# boids[:, 4:6] = 0.1

canvas = scene.SceneCanvas(show=True, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))

color_dict = {"red":(1, 0, 0, 1), "green": (0, 1, 0, 1), "yellow": (1, 1, 0, 1), "white" : (1, 1, 1, 1)}

color_arr = [color_dict["yellow"]]*len(boids)

color_arr[0] = color_dict["red"]

# for i in range(len(color_arr)//2):
#     color_arr[i] = color_dict["green"]

    # [main - нулевая птыца]
    # <|
    #     is_main : 1/0 ,
    #     color: "red",
    #     biods: [1,13.4,3,4]
    #     dt
    # |>
arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=color_arr,
                     # width=5,
                     arrow_size=10,
                     connect='segments',
                     parent=view.scene)


def update(event):
    flocking(boids, perception, coeffs, asp, vrange, better_walls_w, cnt_rely_on = 5)
    propagate(boids, dt, vrange, arange)
    color1 = (0,1,0)
    
    color_arr = [color_dict[random.choice(list(color_dict.keys()))]]*len(boids)
    arrows.arrow_color = color_arr
    arrows.set_data(arrows=directions(boids, dt))
    canvas.update()


if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()
