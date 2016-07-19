#!/usr/bin/env python
# -*- coding: utf-8 -*-

import BaseModel
# for the model simplexlattice, the higher then fourth quarter  is not need .
'''import sympy
from itertools import chain, combinations_with_replacement
m = 1
p = 3
xs = sympy.var(('x{},' * p).format(*range(p)))
tuple(chain.from_iterable(
    map(lambda mi: combinations_with_replacement(xs, mi + 1), range(m))))
y_complete = sympy.Symbol('b0')'''


class SimplexAxial(BaseModel):

    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass
