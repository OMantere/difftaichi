import taichi as ti
import math
import numpy as np
import cv2 
import os
import matplotlib.pyplot as plt 
import time

real = ti.f32
ti.init(default_fp=real)

vis_interval = 1
output_vis_interval = 8 
steps = 2048
vis_resolution = 1024

n_objects = 2
mass_cart = 1
mass_pole = 1
pole_length = 0.3

scalar = lambda: ti.field(dtype=real)
vec = lambda: ti.Vector.field(2, dtype=real)

loss = scalar()
x = scalar()
v = scalar()
a = scalar()

def place():
    ti.root.dense(ti.l, steps).dense(ti.i, n_objects).place(x, v, a)
    ti.root.place(loss)
    ti.root.lazy_grad()

dt = 0.01
learning_rate = 5
g = 9.81

level = 0.25

@ti.kernel
def apply_gravity_force(t: ti.i32):
    xx, th = x[t - 1, 0], x[t - 1, 1]
    vx, vt = v[t - 1, 0], v[t - 1, 1]
    ax, at = a[t - 1, 0], a[t - 1, 1]
    
    F = 0               # This is something we will learn later
    l = pole_length
    mc = mass_cart
    mp = mass_pole
    
    term = (F + mp * l * vt**2 * ti.sin(th)) / (mc + mp)
    bottom_term = l * (4/3 - mp * ti.cos(th)**2 / (mc + mp))
    new_at = (g * ti.sin(th) - ti.cos(th) * term) / bottom_term
    new_ax = term - (mp * l * at * ti.cos(th)) / (mc + mp)
    
    a[t, 0] = new_ax
    a[t, 1] = new_at

@ti.kernel
def time_integrate(t: ti.i32):
    for i in range(n_objects):
        new_v = v[t - 1, i] + dt * a[t, i]
        new_x = x[t - 1, i] + dt * new_v
        v[t, i] = new_v
        x[t, i] = new_x


def visualize(output, t):
    img = np.ones(shape=(vis_resolution, vis_resolution, 3),
                  dtype=np.float32) * (216 / 255.0)

    def circle(x, y, color):
        radius = 0.02
        cv2.circle(img,
                   center=(int(vis_resolution * x),
                           int(vis_resolution * (1 - y))),
                   radius=int(radius * vis_resolution),
                   color=color,
                   thickness=-1)

    # cart
    color = (0.24, 0.3, 0.25)
    circle(x[t, 0], level, color)

    # pole
    def get_pt(x):
        return int(x[0] * vis_resolution), int(vis_resolution -
                                               x[1] * vis_resolution)
    pole_endpoint = [x[t, 0], level]
    pole_endpoint[0] += math.sin(x[t, 1]) * pole_length
    pole_endpoint[1] += math.cos(x[t, 1]) * pole_length
    cv2.line(img,
             get_pt((x[t, 0], level)),
             get_pt(pole_endpoint), (0.2, 0.75, 0.48),
             thickness=4)

    cv2.imshow('img', img)
    cv2.waitKey(1)
    if output:
        cv2.imwrite('cartpole/{}/{:04d}.png'.format(output, t),
                    img * 255)

def forward(output=None):
    interval = vis_interval
    if output:
        interval = output_vis_interval
        os.makedirs('cartpole/{}/'.format(output), exist_ok=True)

    for t in range(1, steps):
        print('step {}'.format(t))
        apply_gravity_force(t)
        time_integrate(t)
        time.sleep(0.05)

        if (t + 1) % interval == 0:
            visualize(output, t)

    #compute_loss(steps - 1)

@ti.kernel
def clear_states():
    for t in range(0, steps):
        for i in range(2):
            x.grad[t, i] = 0
            v.grad[t, i] = 0
            a[t, i] = 0
            a.grad[t, i] = 0

def clear_tensors():
    clear_states()


def main():
    print('Running main')
    place()
    x[0, 0] = 0.5
    x[0, 1] = 0.3

    clear_tensors()
    forward()

if __name__ == '__main__':
    main()
