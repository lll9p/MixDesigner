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
        y = {'y1':63.1,'y2':29.0,'y3':22.2,'y12':50.6,
            'y13':44.5,'y23':26.5,'y123':40.3}
        my_design.formula(**y)
        Or,
        my_design.formula('3d',**y)
        x = {'x1':0.4,'x2':0.5,'x3':0.1}
        my_design.predict(X)
    '''

    def __init__(self, point, lower_bounds=None, upper_bounds=None, interest_area=None):
        self.point = point
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        if lower_bounds is None:
            self.lower_bounds = [0] * point
        if upper_bounds is None:
            self.upper_bounds = [1] * point
        nums = range(self.point)
        self.test_points = tuple(chain.from_iterable(
            map(lambda num: combinations(nums, num + 1), nums)))
        self.M = self._transform_matrix(*self.lower_bounds)
        self.Z = np.array(
            [[1. / len(p) if i in p else 0. for i in range(5)] for p in self.test_points])
        self.X = self.Z.dot(self.M.T)  # at the opposite Z=X*M.T.I
        self._response_surface_coef = None

    @staticmethod
    def _transform_matrix(*args):
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
        return np.array(m)

    def fit(self, y):
        '''
        generate the formula with specific y, y be experiment's results
        assume y's order same as model.test_points
        @useage:
            model.fit(y)
        '''
        if len(y) != len(self.test_points):
            raise TypeError(
                'Missing required positional argument: y\'s length not match test_points')
        # coefficients of response surface
        _response_surface_coef = []
        for i, test_point in enumerate(self.test_points):
            r = len(test_point)
            temp = 0
            for j in range(1, r + 1):
                for test_point_pos in combinations(test_point, j):
                    t = len(test_point_pos)
                    # From 关颖男's 《混料试验设计》 Page:64
                    temp += y[self.test_points.index(test_point_pos)] * \
                        r * (-1)**(r - t) * t**(r - 1)
            _response_surface_coef.append(temp)
        self._response_surface_coef = _response_surface_coef

    def predict(self, X):
        '''
        X is array of arrays
        @useage:
            predict(X)
        '''
        r = []
        for x in X:
            if len(x) != self.point:
                raise TypeError(
                    'Missing required positional argument: x\'s length not match test_points')
            su = 0
            for t, v in zip(self._response_surface_coef, self.test_points):
                for j in v:
                    t *= x[j]
                su += t
            r.append(su)
        return r

    def __str__(self):
        return ('{:+.2f}*{}' * len(self.test_points)).format(
            *chain.from_iterable(
                zip(self._response_surface_coef, [('z_{}*' * len(test_point))
                                                  .format(*map(str, test_point))[:-1] for test_point in self.test_points])))

    def __repr__(self):
        return \
            '''
Point:\t{}
LowerBounds:\t{}
UpperBounds:\t{}
Response surface coef:\t{}
            '''.format(self.point, self.lower_bounds, self.upper_bounds, self.__str__())
