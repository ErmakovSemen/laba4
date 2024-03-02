import imageio as imageio
import numpy as np
import vispy.color
from vispy import app, scene
from vispy.geometry import Rect
from funcs import init_boids, directions, propagate, flocking


app.use_app('pyglet')




w, h = 640, 480
N = 500
dt = 0.1
asp = w / h
perception = 1 / 20
vrange = (0, 0.1)


writer = imageio.get_writer(f'BOIDS_{N}.mp4', fps=60)
c = 0


# ....................c.....a...  s...  w......n
coeffs11 = np.array([0.03, 0.1, 2, 0.05, 0.001]) #жертвы

coeffs22 = np.array([0.01, 0.01, 1, 1, 0.001]) #хищники

coeffs12 = np.array([0.01, 0.01, 20, 0.1, 0.001]) #боятся

coeffs21 = np.array([0.1, 0.1, 3, 0.1, 0.001]) #охотятся

# 0  1   2   3   4   5
# x, y, vx, vy, ax, ay
boids1 = np.zeros((N, 6), dtype=np.float64)
boids2 = np.zeros((N, 6), dtype=np.float64)
init_boids(boids1, asp, vrange=vrange)  # pos and velocity
init_boids(boids2, asp, vrange=vrange)
boids1[:, 4:6] = 0.1  # acceleration
boids2[:, 4:6] = 0.1  # acceleration

canvas = scene.SceneCanvas(show=True, size=(w, h), bgcolor=vispy.color.Color(color=[.12, .12, .12], alpha=1))
view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=Rect(0, 0, asp, 1))

arrows1 = scene.Arrow(arrows=directions(boids1, dt),
                      arrow_color=(0.8, 0.8, 0.8, 1),
                      arrow_size=10,
                      connect='segments',
                      parent=view.scene)
#
arrows2 = scene.Arrow(arrows=directions(boids2, dt),
                      arrow_color=(1, 0, 0, 1),
                      arrow_size=10,
                      connect='segments',
                      parent=view.scene)

# Создание текстовых надписей для отображения FPS и параметров
text_fps = scene.Text(text='FPS:', color='white', pos=(0.1, 0.1), parent=view.scene)
text_N = scene.Text(text='N:', color='white', pos=(.10, .30), parent=view.scene)
text_params = scene.Text(text='Params:', color='white', pos=(.10, .50), parent=view.scene)

text_N.text = f'N1: {N} \n N2: {N} \n '
text_params.text = f'Params: perception={perception}, vrange={vrange} \n Coefficients:\n  1-1: {coeffs11}\n 1-2: {coeffs12}\n 2-1: {coeffs21}\n 2-2: {coeffs22}'

def update(event):
    global c, writer
    if c%2 == 0:
        text_fps.text = f'FPS: {canvas.fps:.2f}'

    flocking(boids1, boids2, perception, coeffs11, coeffs12, coeffs21, coeffs22,  asp, vrange)
    propagate(boids1, boids2, dt, vrange)
    arrows1.set_data(arrows=directions(boids1, dt))
    arrows2.set_data(arrows=directions(boids2, dt))

    if c <= 1800:
        frame = canvas.render(alpha=False)
        writer.append_data(frame)
    else:
        writer.close()
        app.quit()

    c += 1
    print(c)




if __name__ == '__main__':
    input("READY TO RUN ... ")
    timer = app.Timer(interval=0, start=True, connect=update)
    canvas.measure_fps()
    app.run()




