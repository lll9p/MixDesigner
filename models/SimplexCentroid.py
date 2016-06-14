#!/usr/bin/env python
# coding: utf-8
'''
2 to 10 components
'''
import numpy as np
from itertools import combinations, chain


class SimplexCentroid():
    '''
    TODO: should be use helper to help input data, fix the order of y
    @usage:
        my_design = SimplexCentroid(p=3)
        my_design.formula(y1=63.1,y2=29.0,y3=22.2,y12=50.6,y13=44.5,y23=26.5,y123=40.3})
        Or,
        y = {'y1':63.1,'y2':29.0,'y3':22.2,'y12':50.6,'y13':44.5,'y23':26.5,'y123':40.3}
        my_design.formula(**y)
        Or,
        my_design.formula('3d',**y)
        x = {'x1':0.4,'x2':0.5,'x3':0.1}
        my_design.predict(X)
    '''

    def __init__(self, p):
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

    def fit(self, y):
        '''
        generate the formula with specific y, y be experimental results
        @useage:
            y = {'1':v1,'2':v2,'3':v3,...,'123':v123}
            y = {'1': 5, '12': 10, '123': 13, '13': 2, '2': 11, '23': 10, '3': 8}
            make_yf(y)
        '''
        y = {k: y[''.join(map(str, k))] for k in self.base_arr}
        if len(self.base_arr) != len(y):
            raise TypeError(
                'Missing required positional argument: not enugh y')
        self.yf = tuple(sum(self._ftree[k][yk] * y[yk]
                            for yk in self._ftree[k]) for k in self.base_arr)
        return self.yf

    def predict(self, X):
        '''
        same as self.value, but is the list version
        caculate the value with specific X
        X is array-like
        @useage:
            predict(X)
        '''
        r = []
        for x in X:
            if len(x)!= self.p:
                raise TypeError(
                    'Missing required positional argument: not enough x')
            su = 0
            for t, v in zip(self.yf, self.base_arr):
                for j in v:
                    t *= x[j - 1]
                su += t
            r.append(su)
        return r


class SimplexCentroidLowerConstraints(SimplexCentroid):

    def __init__(self, p, bounds):
        SimplexCentroid.__init__(self, p)
        self.Z = self.transform_matrix(*bounds)

    def z(self, **args):
        '''
        z = Z*x.T
        return real value of x
        @useage:
            value_real(x1=0,x2=0,x3=0)
        '''
        if len(args) != self.p:
            raise TypeError(
                'Missing required positional argument: not enough x')
        if not np.isclose(sum(np.abs(tuple(args.values()))), 1.0):
            raise ValueError(
                'Sumutation of x should be 1, and x should be positive')
        return self.Z * np.matrix(tuple(args[i] for i in sorted(args))).T

    @staticmethod
    def transform_matrix(*args):
        '''
        @useage:
            transform_matrix(x1_bound,x2_bound...)
        m.dot(Z.T)
        '''
        m = []
        p = len(args)
        s = 1 - sum(args)
        for i, a in enumerate(args):
            m.append([a] * p)
            m[-1][i] += s
        return np.matrix(m)
