#!/usr/bin/env python
# coding: utf-8
import collections
from itertools import chain, combinations

import numpy as np


# import ternary
# All calculation will be in Z, not the real value X.


class SimplexCentroid():
    '''
    SimplexCentroid model
    @usage:
    model = SimplexCentroid(5,[0.6,0,0,0,0])
    model.fit(y)
    model.predict(X)
    '''

    def __init__(self, n_point, lower_bounds=None, upper_bounds=None):
        self.n_point = n_point
        self.lower_bounds = np.array(
            lower_bounds if lower_bounds else [0] * self.n_point)
        self.upper_bounds = np.array(
            upper_bounds if upper_bounds else [1] * self.n_point)
        nums = range(self.n_point)
        # generate a powerset but void set
        self.test_points = tuple(chain.from_iterable(
            map(lambda num: combinations(nums, num + 1), nums)))
        # transform_matrix
        self._M = self.lower_bounds.repeat(self.n_point).reshape(
            (self.n_point, self.n_point)) \
            + np.eye(self.n_point) \
            * (1 - self.lower_bounds.sum())
        self._Z = np.array(
            [
                [1. / len(p) if i in p else 0. for i in range(self.n_point)]
                for p in self.test_points
            ]
        )
        # Z=X*M.T.I
        self._X = self._Z.dot(self._M.T)
        self._response_surface_coef = None

    def X2Z(self, X):
        return X.dot(self._M.T.I)

    def Z2X(self, Z):
        return Z.dot(self._M.T)

    def fit(self, y):
        '''
        generate the formula with specific y, y be experiment's results
        assume y's order same as model.test_points
        @useage:
        model.fit(y)
        '''
        if len(y) != len(self.test_points):
            raise TypeError(
                'Missing required positional argument: \
                y\'s length not match test_points')
        ZZ = np.array([self._Z.take(test_point_pos,axis=1).prod(axis=1) \
                       for test_point_pos in self.test_points]).T
        _response_surface_coef = np.linalg.solve(ZZ,y)
        self._response_surface_coef = _response_surface_coef
        return self._response_surface_coef
        # coefficients of response surface
        #_response_surface_coef = []
        #for i, test_point in enumerate(self.test_points):
        #    r = len(test_point)
        #    temp = 0
        #    for j in range(1, r + 1):
        #        for test_point_pos in combinations(test_point, j):
        #            t = len(test_point_pos)
        #            # From 关颖男's 《混料试验设计》 Page:64
        #            temp += y[self.test_points.index(test_point_pos)] * \
        #                r * (-1)**(r - t) * t**(r - 1)
        #    _response_surface_coef.append(temp)
        #self._response_surface_coef = np.array(_response_surface_coef)

    def predict(self, X):
        '''
        X is array of arrays, x is the read value, not code value
        @useage:
        model.predict(X)
        '''
        notarray = any(map(lambda x: not isinstance(
            x, (collections.Sequence, np.ndarray)), X))
        shape = (1 if notarray else len(X), self.n_point)
        try:
            X = np.reshape(X, shape)
            if X.ndim != 2:
                raise TypeError(
                    'X is not a valid array-like object!')
            if X.shape != shape:
                raise TypeError(
                    'Missing required positional argument: \
                    x\'s length not match test_points')
            if X.dtype != np.float:
                raise TypeError(
                    'DataType of element(s) of X is wrong! Please check again.')
        except:
            raise TypeError(
                'DataType of element(s) of X is wrong! Please check again.')
        XX = X.dot(np.linalg.inv(self._M.T))
        # from each XX, take the points and make a prod,
        # then multiply coef and then sums up
        prediction = self._response_surface_coef.dot(
            [XX.take(test_point_pos, axis=1).prod(axis=1)
             for test_point_pos in self.test_points]
        )
        return prediction

    def score(self, X, y):
        return np.sum(np.abs(self.predict(X) - y)) / len(y)

    def plot(self, side):
        '''
        side = [0,1,2] means choose x0,x1,x2 to draw
        '''
        points = np.mgrid[0:101, 0:101, 0:101].reshape(3, -1)
        mesh_points = points[:, np.where(points.sum(axis=0) == 100)[0]]
        X_points = np.zeros((self.n_point, mesh_points.shape[1]), dtype=int)
        X_points[side, :] = mesh_points
        Y = self.predict(X_points.T / 100.0)
        data = {tuple(points): y for points, y in zip(mesh_points.T, Y)}
        ternary.heatmap(data, scale=100)

    def __str__(self):
        if self._response_surface_coef is None:
            model_str = ''
        else:
            # ugly code NEED reform
            model_str = ('{:+.2f}*{}' * len(self.test_points)).format(
                *chain.from_iterable(
                    zip(self._response_surface_coef,
                        [('z_{}*' * len(test_point))
                         .format(*map(str, test_point))[:-1]
                         for test_point in self.test_points])
                )
            )
        return model_str

    def __repr__(self):
        return \
            '''
            Point:\t{}
            LowerBounds:\t{}
            UpperBounds:\t{}
            Response surface coef:\t{}
            '''.format(self.n_point,
                       self.lower_bounds,
                       self.upper_bounds,
                       self.__str__()
                       )
