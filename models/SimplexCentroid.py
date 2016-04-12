#!/usr/bin/env python
# coding: utf-8
'''
2 to 10 components
'''
import numpy as np
import operator
from itertools import combinations,chain
from functools import reduce

class SimplexCentroid():
    '''
    @usage:
        my_design = simplex_center_designer(p=3)
        
        
        my_design.formula(yname='3d',y1=63.1,y2=29.0,y3=22.2,y12=50.6,y13=44.5,y23=26.5,y123=40.3})
        Or,
        y = {'y1':63.1,'y2':29.0,'y3':22.2,'y12':50.6,'y13':44.5,'y23':26.5,'y123':40.3}
        my_design.formula(yname='3d',**y)
        Or,
        my_design.formula('3d',**y)
        
        
        x = {'x1':0.4,'x2':0.5,'x3':0.1}
        my_design.value(yname='3d',**x)
    '''

    def __init__(self, p):
        self.p = p
        self.yf = dict()
        self.vf = dict()
        nums = range(1, self.p + 1)
        self.base_arr = tuple(chain.from_iterable(map(lambda num:combinations(nums,num),nums)))
        #self.base_arr = tuple(combines for num in nums for combines in combinations(nums, num))
        self.ftree = self.make_ftree()

    def make_ftree(self):
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
        self.yf[yname] = tuple(sum(self.ftree[k][yk] * y[yk]
                            for yk in self.ftree[k]) for k in self.base_arr)
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
        if not np.isclose(sum(np.abs(x)), 1.0,rtol=1e-2):
            raise ValueError(
                'Sumutation of x should be 1, and x should be positive')
        su = 0
        for t,v in zip(self.yf[yname],self.base_arr):
            for j in v:
                t*=x[j-1]
            su+=t
        return su
        '''#a little bit slower method
        return sum((
            a * b for a, b in zip(
                (reduce(operator.mul, (x[i - 1] for i in xx))
                 for xx in self.base_arr),
                (self.yf[yname])
            )))
        '''
    '''
    def plot2D(self, yname, **args):
        f = partial(self.value, yname)
        return ternary.plot_tri_contourf(f, **args) #
    '''


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
        if len(args) != len(self.x):
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
        '''
        m = []
        p = len(args)
        s = 1 - sum(args)
        for i, a in enumerate(args):
            m.append([a] * p)
            m[-1][i] += s
        return np.matrix(m)

if __name__ == "__main__":
    spices_design2 = simplex_center_designer_bounded(p=3, bounds=[.2, .4, .2])
    y2 = {'1': 5, '2': 11, '3': 8, '12': 10, '13': 2, '23': 10, '123': 13}
    spices_design2.fit('test1', y2)
    fig = spices_design.plot2D('test1',nlevels=200, subdiv=8, cmap=plt.cm.inferno)

