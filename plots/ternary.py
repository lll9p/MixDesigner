#!/usr/bin/env python
# coding: utf-8
# my ternary implement based on
# http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import math
# import seaborn as sns
# sns.set_context('notebook',)
# sns.set(context='notebook',
#         style='ticks',
#         font='WenQuanYi Micro Hei',
#         font_scale=1.2)
matplotlib.use('TkAgg')  # maybe not
corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])  # cos(30)
triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
# Mid-points of triangle sides opposite of each corner
midpoints = [(corners[(i + 1) % 3] + corners[(i + 2) % 3]) / 2.0
             for i in range(3)]


def shannon_entropy(p):
    """Computes the Shannon Entropy at a distribution in the simplex."""
    s = 0.
    for i in range(len(p)):
        try:
            s += p[i] * math.log(p[i])
        except ValueError:
            continue
    return -1. * s


def xy2bc(xy, tol=1.e-3):
    '''Converts 2D Cartesian coordinates to barycentric.
    笛卡尔坐标转重心坐标'''
    s = [(corners[i] - midpoints[i]).dot(xy - midpoints[i]) / 0.75
         for i in range(3)]
    return np.clip(s, tol, 1.0 - tol)


def tick_txy(location, width=1.0, size=20):
    height = width * 0.75**0.5
    if location == 'left':
        xy = np.array((
            np.arange(0, width / 2 / size * (size + 1), width / 2 / size),
            np.arange(0, height / size * (size + 1), height / size), ))
        return xy[0, :][::-1], xy[1, :][::-1]
    if location == 'right':
        xy = np.array((
            np.arange(0.5, 0.5 + width / 2 / size * (size + 1),
                      width / 2 / size),
            np.arange(0, height / size * (size + 1), height / size), ))
        return xy[0, :][::-1], xy[1, :]
    if location == 'bottom':
        xy = np.array((
            np.arange(0, width / size * (size + 1), width / size),
            np.array((0.0, ) * (size + 1)), ))
        return xy[0, :], xy[1, :]


'''
http://stackoverflow.com/questions/29512046/how-to-create-ternary-contour-plot-in-python
http://stackoverflow.com/questions/30967849/python-matplotlib-tick-marks-on-ternary-plot

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np

def plot_ticks(start, stop, tick, n):
    r = np.linspace(0, 1, n+1)
    x = start[0] * (1 - r) + stop[0] * r
    x = np.vstack((x, x + tick[0]))
    y = start[1] * (1 - r) + stop[1] * r
    y = np.vstack((y, y + tick[1]))
    plt.plot(x, y, 'k', lw=1)

n = 10
tick_size = 0.2
margin = 0.05

# define corners of triangle
left = np.r_[0, 0]
right = np.r_[1, 0]
top = np.r_[0.5, 3**0.5 / 2]
triangle = np.c_[left, right, top, left]

# define vectors for ticks
bottom_tick = tick_size * (right - top) / n
right_tick = tick_size * (top - left) / n
left_tick = tick_size * (left - right) / n

plt.plot(triangle[0], triangle[1], 'k', lw=2)
plot_ticks(left, right, bottom_tick, n)
plot_ticks(right, top, right_tick, n)
plot_ticks(left, top, left_tick, n)
plt.axis([left[0]-margin, right[0]+margin, left[1]-margin, top[1]+margin])
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
'''


def tick_labels(scale=100, size=20):
    return [
        str(int(i))
        for i in np.arange(0, scale / size * (size + 1), scale / size)
    ]


def plot_tri_contourf(distribute_func, n_levels=200, subdiv=8, **kwargs):
    # scale, zoom, Xnames=('x1', 'x2', 'x3'), size=(10, 8),
    # fontsize=20, **args
    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    pvals = [distribute_func(xy2bc(xy)) for xy in zip(trimesh.x, trimesh.y)]
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.triplot(triangle, color='black')
    trimap = ax.tricontourf(trimesh, pvals, n_levels, **kwargs)
    # trimap = ax.tricontourf(x,y,t,n_levels,**kwargs)
    offset = 0.02
    linewidth = 1.
    for x, y, s in zip(*tick_txy('left'), tick_labels()):
        ax.text(
            x - offset / 2,
            y + 0.75**0.5 * offset,
            s,
            verticalalignment='center',
            horizontalalignment='right')
        ax.plot(
            (x, x - offset / 2), (y, y + 0.75**0.5 * offset),
            '-',
            lw=linewidth,
            color='black')
    for x, y, s in zip(*tick_txy('right'), tick_labels()):
        ax.text(
            x + offset / 2,
            y,
            s,
            verticalalignment='center',
            horizontalalignment='left')
        ax.plot((x, x + offset / 2), (y, y), '-', lw=linewidth, color='black')
    for x, y, s in zip(*tick_txy('bottom'), tick_labels()):
        ax.text(
            x - offset / 2,
            y - 0.75**0.5 * offset,
            s,
            verticalalignment='top',
            horizontalalignment='center')
        ax.plot(
            (x, x - offset / 2), (y, y - 0.75**0.5 * offset),
            '-',
            lw=linewidth,
            color='black')
    '''
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(plt.gca())
    cax = divider.append_axes("right", "5%", pad="-8%")
    fig.colorbar(trimap, cax=cax)
    '''
    fig.colorbar(trimap)
    ax.set_aspect('equal')
    fig.tight_layout(pad=0)
    ax.axis('equal')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(-0.02, 0.75**0.5+0.02)
    ax.axis('off')
    return fig
