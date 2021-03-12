#!/usr/bin/env python
# coding: utf-8
# my ternary implement based on
# http://blog.bogatron.net/blog/2014/02/02/visualizing-dirichlet-distributions/
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import math
import matplotlib.lines as mlines


def shannon_entropy(pp):
    """Computes the Shannon Entropy at a distribution in the simplex."""
    r = []
    for p in pp:
        s = 0.
        for i in range(len(p)):
            try:
                s += p[i] * math.log(p[i])
            except ValueError:
                continue
        r.append(-1. * s)
    return r


def plot_ternary(distribute_func, n_levels=200, subdiv=8, **kwargs):
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])  # cos(30)
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])
# Mid-points of triangle sides opposite of each corner

    RI = np.linalg.inv(np.vstack((corners.T, [1, 1, 1])))

    def xy2bc(xys, tol=1.e-3):
        '''
        Converts 2D Cartesian coordinates to barycentric.
        according to https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Conversion_between_barycentric_and_Cartesian_coordinates'''
        xysT = np.transpose(xys)
        ones = [1] * len(xys)
        xysT1 = np.vstack((xysT, ones))
        lambda_ = RI.dot(xysT1).T
        return np.clip(lambda_, a_min=tol, a_max=1.0 - tol)

    def tick_labels(scale=100, size=20):
        return [
            str(int(i))
            for i in np.arange(0, scale / size * (size + 1), scale / size)
        ]

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

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    bc = xy2bc(list(zip(trimesh.x, trimesh.y)))
    pvals = distribute_func(bc)
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


def rotate(degree):
    def degree2rad(degree):
        return np.pi * degree / 180.0
    rad = degree2rad(degree)
    M = np.array([
        [np.cos(rad), -np.sin(rad)],
        [np.sin(rad), np.cos(rad)]
    ])
    return M


def pl(distribute_func, n_levels=200, subdiv=8, scale=[[0., 1.], [0., 1.], [0., 1.]]):
    corners = np.array([[0, 0], [1, 0], [0.5, 0.75**0.5]])  # cos(30)
    triangle = tri.Triangulation(corners[:, 0], corners[:, 1])

    RI = np.linalg.inv(np.vstack((corners.T, [1, 1, 1])))

    def xy2bc(xys, tol=1.e-4, RI=RI):
        '''
        Converts 2D Cartesian coordinates to barycentric.
        According to https://en.wikipedia.org/wiki/Barycentric_coordinate_system#Conversion_between_barycentric_and_Cartesian_coordinates'''
        xysT = np.transpose(xys)
        ones = [1] * len(xys)
        xysT_ = np.vstack((xysT, ones))
        lambda_ = RI.dot(xysT_).T
        return np.clip(lambda_, a_min=tol, a_max=1.0 - tol)

    def ticktext(scale=scale):
        scale = np.array(scale)
        if scale.shape != (3, 2):
            raise "scale's shape {} is not right!".format(scale.shape)
        sa, sb, sc = scale  # scale a, b, and c.

    def ticklines(tickscale=[0, 0.75**0.5], split=10, line_height=0.1, clockwise=60):
        scale_butt, scale_top = tickscale
        step_y = (scale_top - scale_butt) / split
        init_tick_location_y = np.arange(
            scale_butt, scale_top + step, step)
        step_x = (1. - 0.) / split
        init_tick_location_x = init_tick_location_y / (3**0.5)
        np.flipud(init_tick_location_x)

    refiner = tri.UniformTriRefiner(triangle)
    trimesh = refiner.refine_triangulation(subdiv=subdiv)
    bc = xy2bc(list(zip(trimesh.x, trimesh.y)))
    pvals = distribute_func(bc)
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    ax.triplot(triangle, color='black')
    trimap = ax.tricontourf(trimesh, pvals, n_levels)
    fig.colorbar(trimap)
    mlines.Line2D([], [])
    ax.set_aspect('equal')
    fig.tight_layout(pad=0)
    ax.axis('equal')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(-0.02, 0.75**0.5+0.02)
    ax.axis('off')
    return fig, ax
