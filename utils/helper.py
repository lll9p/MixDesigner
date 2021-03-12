#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def input_y_helper(base_arr, flag=False):
    ys = [''.join([str(n) for n in arr]) for arr in base_arr]
    if flag:
        y = [float(i) for i in input('Input the y array pls.\n').split('\n')]
    else:
        y = [float(input('Input y_{}:'.format(yi))) for yi in ys]
    return dict(zip(ys, y))


def coded_helper(model):
    p = model.point
    base_arr = model.base_arr
    r = []
    for l in base_arr:
        t = [0] * p
        avg = 1 / len(l)
        for i in l:
            t[i - 1] = avg
        r.append(t)
    return r


def mixture_proportion_helper(model):
    '''
    help design the proportions
    TODO:
        reverse the processing
    '''
    m = coded_helper(model)
    try:
        Z = model.Z
    except:
        Z = None
    return np.matrix(m) * Z.T
