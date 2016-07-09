#!/usr/bin/env python
# coding: utf-8
import numpy as np
from itertools import combinations, chain
from functools import reduce


class SimplexCentroid():
    '''
    TODO: should be use helper to help input data, fix the order of y
    @usage:
        model = SimplexCentroid(5,[0.6,0,0,0,0)
        model.fit(y)
        model.predict(X)
    '''

    def __init__(self, point, lower_bounds=None, upper_bounds=None):
        self.point = point
        self.lower_bounds = np.array(
            lower_bounds if lower_bounds else [0] * self.point)
        self.upper_bounds = np.array(
            upper_bounds if upper_bounds else [1] * self.point)
        nums = range(self.point)
        self.test_points = tuple(chain.from_iterable(
            map(lambda num: combinations(nums, num + 1), nums)))
        # transform_matrix
        self._M = self.lower_bounds.repeat(self.point).reshape(
            (self.point, self.point)) + np.eye(self.point) * (1 - self.lower_bounds.sum())
        self._Z = np.array(
            [[1. / len(p) if i in p else 0. for i in range(self.point)] for p in self.test_points])
        self._X = self._Z.dot(self._M.T)  # at the opposite Z=X*M.T.I
        self._response_surface_coef = None

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
            model.predict(X)
        '''
        try:
            X_ = np.array(X)
            if X_.ndim != 2:
                if X_.ndim == 1:
                    X_ = X_.reshape((1, self.point))
            Z = X_.dot(np.linalg.inv(self._M.T))
        except:
            raise TypeError(
                'X is not a valid array-like object!')

        if Z.shape[1] != self.point:
            raise TypeError(
                'Missing required positional argument: x\'s length not match test_points')
        prediction = np.apply_along_axis(lambda x: sum(reduce(lambda a, b: a * b, x.take(test_point_pos)) *
                                                       coef for coef, test_point_pos in zip(self._response_surface_coef, self.test_points)), 1, Z)
        return prediction

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
