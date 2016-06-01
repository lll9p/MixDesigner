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
        列出多项式，用混料成分代替之
        tree = dict()
        for k in self.base_arr:
            r = len(k)
            tree[k] = {}
            for j in range(1, r + 1):
                for coefk in combinations(k, j):
                    t = len(coefk)
                    tree[k].update({coefk: r * (-1)**(r - t) * t**(r - 1)})
        return tree
        '''
        pass

    def fit(self, yname, y):
        '''
        generate the formula with specific y, y be experimental results
        @useage:
            y = {'1':v1,'2':v2,'3':v3,...,'123':v123}
            y = {'1': 5, '12': 10, '123': 13, '13': 2, '2': 11, '23': 10, '3': 8}
            make_yf(yname='test1',y)
        '''
        y = {k: y[''.join(map(str, k))] for k in self.base_arr}
        if len(self.base_arr) != len(y):
            raise TypeError(
                'Missing required positional argument: not enugh y')
        self.yf[yname] = tuple(sum(self._ftree[k][yk] * y[yk]
                                   for yk in self._ftree[k]) for k in self.base_arr)
        return self.yf[yname]

    def predict(self, yname, x):
        '''
        same as self.value, but is the list version
        caculate the value with specific x
        @useage:
            value('test',(1,0,0))
        '''
        if len(x) != self.p:
            raise TypeError(
                'Missing required positional argument: not enough x')
        if not np.isclose(sum(np.abs(x)), 1.0, rtol=1e-2):
            raise ValueError(
                'Sumutation of x should be 1, and x should be positive')
        su = 0
        for t, v in zip(self.yf[yname], self.base_arr):
            for j in v:
                t *= x[j - 1]
            su += t
        return su

