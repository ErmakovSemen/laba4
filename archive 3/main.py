import numpy as np
import random
from vispy import app, scene
from vispy.geometry import Rect
import imageio as imageio

from funcs import init_boids, directions, propagate, flocking, periodic_walls, wall_avoidance
app.use_app('PyQt5')

writer = imageio.get_writer('boids5000.mp4', fps=60)

frame = 0
w, h = 1500, 800
N = 5000
dt = 0.1
asp = w / h
perception = 1/20
# walls_order = 8
better_walls_w = 0.05
vrange=(0, 0.1)
arange=(0, 0.01)
# cnt_in_a_view = 10
cnt_rely_on = 50
#                    c      a    s      w
coeffs = np.array([0.88, 0.5,  0.9,  0.03])

# 0  1   2   3   4   5
# x, y, vx, vy, ax, ay
boids = np.zeros((N, 6), dtype=np.float64)
init_boids(boids, asp, vrange=vrange)
# boids[:, 4:6] = 0.1

canvas = scene.SceneCanvas(show=True, size=(w, h))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))

color_dict = {"red":(1, 0, 0, 1), "green": (0, 1, 0, 1), "yellow": (1, 1, 0, 1), "white" : (1, 1, 1, 1)}

color_arr = [color_dict["yellow"]]*N

color_arr[0] = color_dict["red"]

arrows = scene.Arrow(arrows=directions(boids, dt),
                     arrow_color=color_arr,
                     # width=5,
                     arrow_size=10,
                     connect='segments',
                     parent=view.scene)

canvas.title = "Flocking simulation"
text_fps = scene.Text(text='FPS:', color='white', pos=(0.1, 0.1), parent=view.scene)
text_basic = scene.Text(text=f'Birds: {N}', color='white', pos=(0.1, 0.5), parent=view.scene)
text_basic = scene.Text(text=f'coefs: {coeffs}', color='white', pos=(0.1, 0.3), parent=view.scene)
text_basic = scene.Text(text=f'cnt_rely_on: {cnt_rely_on}', color='white', pos=(0.1, 0.4), parent=view.scene)

def update(event):
    global frame
    calculated_data = flocking(boids, perception, coeffs, asp, vrange, better_walls_w, cnt_rely_on = cnt_rely_on)
    propagate(boids, dt, vrange, arange)
    # periodic_walls(boids, asp)
    wall_avoidance(boids, asp)
    print(calculated_data["mask_see"].shape)
    color_arr = np.array([color_dict["white"]]*N)
    

    color_arr[calculated_data["mask_see"][0]] =  color_dict["green"]
    color_arr[calculated_data["mask_rely"][0]] =  color_dict["yellow"]
    
    # color_arr = [color_dict[random.choice(list(color_dict.keys()))]]*len(boids)
    color_arr[0] = color_dict["red"] # recolor a leader

    arrows.arrow_color = color_arr #apply colors
    arrows.set_data(arrows=directions(boids, dt))
    text_fps.text = f'FPS: {canvas.fps:.2f}'

    if frame < 1800:

        cadr = canvas.render(alpha=False)
        writer.append_data(cadr)
        
        # canvas.update()

    else:
        writer.append_data(cadr)
        writer.close()
        app.quit()
        
    frame += 1
    



if __name__ == '__main__':
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()

