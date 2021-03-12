#!/usr/bin/env python
# coding: utf-8
import numpy as np
from itertools import combinations, combinations_with_replacement, chain
# for the model simplexlattice, the higher then fourth quarter is not need .
'''
import sympy
from itertools import combinations_with_replacement
m = 1
p = 3
xs = sympy.var(('x{},' * p).format(*range(p)))
tuple(chain.from_iterable(
    map(lambda mi: combinations_with_replacement(xs, mi + 1), range(m))))
y_complete = sympy.Symbol('b0')
'''


def canonical_polynomial_model_test_points(point, degree):
    '''
    return response_model,values_model
    '''
    def _linear(point):
        '''
        degree = 1
        '''
        return [list(i) for i in range(point)]

    def _quadratic(point):
        '''
        degree = 2
        '''
        return [list(i) for i in range(point)] + \
            list(combinations_with_replacement(range(point), 2))

    def _full_cubic(point):
        '''
        degree = 3
        '''
        return [list(i) for i in range(point)] + \
            list(combinations_with_replacement(range(point), 2)) + \
            list(combinations_with_replacement(range(point), 3))
        # seems this is the wrong way to do it ...

    def _special_cubic(point):
        '''
        degree = 4
        '''
        pass
    model_dict = {1: _linear,
                  2: _quadratic,
                  3: _full_cubic,
                  4: _special_cubic,
                  }
    return model_dict.get(degree)(point)
    # y dict is better to user sympy

class SimplexLattice():
    '''
    SimplexLattice model
    This model's split should be less then 4.
    @usage:
        model = SimplexLattice(5,[0.6,0,0,0,0])
        model.fit(y)
        model.predict(X)
    '''

    def __init__(self, point, degree, lower_bounds=None, upper_bounds=None):
        self.point = point
        self.degree = degree
        self.lower_bounds = np.array(
            lower_bounds if lower_bounds else [0] * self.point)
        self.upper_bounds = np.array(
            upper_bounds if upper_bounds else [1] * self.point)
        nums = range(self.point)
        self.test_points = tuple(chain.from_iterable(
            map(lambda num: combinations(nums, num + 1), nums)))
        # transform_matrix
        self._M = self.lower_bounds.repeat(self.point).reshape(
            (self.point, self.point)) \
            + np.eye(self.point) \
            * (1 - self.lower_bounds.sum())
        self._Z = np.array(
            [
                [1. / len(p) if i in p else 0. for i in range(self.point)]
                for p in self.test_points
            ]
        )
        # Z=X*M.T.I
        self._X = self._Z.dot(self._M.T)
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
                'Missing required positional argument: \
                y\'s length not match test_points')
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
        self._response_surface_coef = np.array(_response_surface_coef)

    def predict(self, X):
        '''
        X is array of arrays
        @useage:
            model.predict(X)
        '''
        try:
            if type(X) != np.ndarray:
                X = np.array(X)
            if X.ndim != 2:
                if X.ndim == 1:
                    X = X.reshape((1, self.point))
            Z = X.dot(np.linalg.inv(self._M.T))
        except:
            raise TypeError(
                'X is not a valid array-like object!')
        if Z.shape[1] != self.point:
            raise TypeError(
                'Missing required positional argument: \
                x\'s length not match test_points')
        # from each Z, take the points and make a prod,
        # then multiply coef and sums up
        prediction = self._response_surface_coef.dot(
            [Z.take(test_point_pos, 1).prod(1)
             for test_point_pos in self.test_points]
        )
        return prediction

    def __str__(self):
        if not self._response_surface_coef:
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
            '''.format(self.point,
                       self.lower_bounds,
                       self.upper_bounds,
                       self.__str__()
                       )
