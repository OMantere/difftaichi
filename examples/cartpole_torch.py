import math
import time
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

mc = 1
mp = 0.1
l = 0.5
g = 9.81
dt = 0.01
max_cart_force = 10

level = 0.2
vis_resolution = 1024
iterations = 10000
steps = 200
learning_rate = 0.03


class Controller(nn.Module):
    def __init__(self):
        super(Controller, self).__init__()
        self.l1 = nn.Linear(4, 24)
        self.l2 = nn.Linear(24, 48)
        self.l3 = nn.Linear(48, 1)
    def forward(self, xx, xt, vx, vt, ax, at):
        x = torch.stack((xx, xt, vx, vt))
        x = torch.relu(self.l1(x))
        x = torch.relu(self.l2(x))
        x = torch.tanh(self.l3(x))
        return max_cart_force * x[0]


def forward(state, net):
    xx, xt, vx, vt, ax, at = state
    F = net.forward(xx, xt, vx, vt, ax, at)

    term = (F + mp * l * vt**2 * torch.sin(xt)) / (mc + mp)
    bottom_term = l * (4/3 - mp * torch.cos(xt)**2 / (mc + mp))
    new_at = (g * torch.sin(xt) - torch.cos(xt) * term) / bottom_term
    new_ax = term - (mp * l * at * torch.cos(xt)) / (mc + mp)

   # integration
    new_vt = vt + dt * new_at
    new_xt = xt + dt * new_vt
    new_vx = vx + dt * new_ax
    new_xx = xx + dt * new_vx

    return [new_xx, new_xt, new_vx, new_vt, new_ax, new_at]


def visualize(state, output=None):
    xx, xt, vx, vt, ax, at = state
    x_pos = xx.item()

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
    circle(x_pos, level, color)

    # pole
    def get_pt(x):
        return int(x[0] * vis_resolution), int(vis_resolution -
                                               x[1] * vis_resolution)
    pole_endpoint = [x_pos, level]
    pole_endpoint[0] += math.sin(xt.item()) * l
    pole_endpoint[1] += math.cos(xt.item()) * l
    cv2.line(img,
         get_pt([x_pos, level]),
         get_pt(pole_endpoint), (0.2, 0.75, 0.48),
         thickness=4)

    cv2.imshow('img', img)
    cv2.waitKey(1)
    if output:
        cv2.imwrite('cartpole/{}/{:04d}.png'.format(output, t),
                    img * 255)



def main():
    print("running main")

    net = Controller().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=learning_rate)
 
    theta_threshold = 24 * np.pi / 360
    x_threshold = 0.4
    ZERO = torch.zeros([]).to(device)
    def loss_fn(state):
        xx, xt, vx, vt, ax, at = state
        x_upper = x_threshold - (xx - starting_xx)
        x_lower = x_threshold + (xx - starting_xx)
        r_x = torch.max(ZERO, torch.min(x_upper, x_lower))

        t_upper = theta_threshold - xt
        t_lower = theta_threshold + xt
        r_t = torch.max(ZERO, torch.min(t_upper, t_lower))

        return (r_t - theta_threshold)**2 + 0.01 * (r_x - x_threshold)**2

    for i in range(iterations):
        t0 = time.time()

        state = list(torch.zeros([]).to(device) for i in range(6))
        starting_theta = np.random.rand() * 0.1 - 0.05
        starting_xx = 0.5
        with torch.no_grad():
            state[1] += starting_theta
            state[0] += starting_xx

        max_seq = 8
        seq = 1
        losses = []
        for s in range(steps):
            visualize(state)
            state = forward(state, net)

            if state[1]**2 > (2 * theta_threshold)**2:
                break
            if (state[0]-starting_xx)**2 > x_threshold**2:
                break

            losses.append(loss_fn(state) * 1000)
            if not seq % max_seq:
                loss = torch.mean(torch.stack(losses))
                print("iteration {} steps {} time {:.4f} loss {:.4f}".format(i, s, time.time() - t0, loss.item()))
                opt.zero_grad()
                loss.backward()
                opt.step()
                for si in range(len(state)):
                    state[si] = state[si].data
                losses = []
            seq += 1

        # with torch.no_grad():
            # for p in net.parameters():
                # print(p.grad.norm())
                # p-= learning_rate * p.grad


if __name__ == '__main__':
    main()
