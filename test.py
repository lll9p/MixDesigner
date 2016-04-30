from plots import ternary as tr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn;sn.set(font='WenQUanYi Micro Hei')
matplotlib.use('TkAgg')
from models import SimplexCentroid as sc
from functools import partial
spices_design = sc.SimplexCentroid_bounded(p=3, bounds=[.2, .4, .2])
y2 = {'1': 5, '2': 11, '3': 8, '12': 10, '13': 2, '23': 10, '123': 13}
spices_design.make_yf('test1', y2)
tr.plot_tri_contourf(partial(spices_design.value,'test1'),nlevels=200, subdiv=8, cmap='inferno')
if __name__ == "__main__":
    spices_design2 = simplex_center_designer_bounded(p=3, bounds=[.2, .4, .2])
    y2 = {'1': 5, '2': 11, '3': 8, '12': 10, '13': 2, '23': 10, '123': 13}
    spices_design2.fit('test1', y2)
    fig = spices_design.plot2D('test1',nlevels=200, subdiv=8, cmap=plt.cm.inferno)

