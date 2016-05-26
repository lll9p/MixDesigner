#!/usr/bin/env python
# -*- coding: utf-8 -*-


def input_y_helper(base_arr):
    ydict = {}
    for arr in base_arr:
        y = 'y' + ''.join([str(n) for n in arr])
        ydict[y] = input('Input {}:'.format(y))
    return ydict
