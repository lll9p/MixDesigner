#!/usr/bin/env python
'''
2 to 20 components
'''
import numpy as np
from itertools import combinations, chain


class SimplexLattice():
    '''
    @usage:
        my_design = SimplexLattice(p=3)


        my_design.formula(yname='3d',y1=63.1,y2=29.0,y3=22.2,y12=50.6,y13=44.5,y23=26.5,y123=40.3})
        Or,
        y = {'y1':63.1,'y2':29.0,'y3':22.2,'y12':50.6,'y13':44.5,'y23':26.5,'y123':40.3}
        my_design.formula(yname='3d',**y)
        Or,
        my_design.formula('3d',**y)


        x = {'x1':0.4,'x2':0.5,'x3':0.1}
        my_design.value(yname='3d',**x)
    '''

    def __init__(self, m=2, p=3):
        self.p = p
        self.yf = dict()
        self.vf = dict()
        nums = range(1, self.p + 1)
        self.base_arr = tuple(chain.from_iterable(
            map(lambda num: combinations(nums, num), nums)))
        self._ftree = self._make_ftree()

    def _make_ftree(self):
        '''
        '''
        tree = dict()
        for k in self.base_arr:
            r = len(k)
            tree[k] = {}
            for j in range(1, r + 1):
                for coefk in combinations(k, j):
                    t = len(coefk)
                    tree[k].update({coefk: r * (-1)**(r - t) * t**(r - 1)})
        return tree
