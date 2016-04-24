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

    def __init__(self, p=3):

        pass
