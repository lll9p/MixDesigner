#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def input_y_helper(base_arr):
    ydict = {}
    for arr in base_arr:
        y = 'y' + ''.join([str(n) for n in arr])
        ydict[y] = input('Input {}:'.format(y))
    return ydict


def coded_helper(model):
    p = model.p
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
    m = coded_helper(model)
    try:
        Z = model.Z
    except:
        Z = None
    return np.matrix(m) * Z.T
