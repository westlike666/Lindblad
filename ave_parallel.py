# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 16:18:30 2022

@author: westl
"""

import numpy as np
import os
import utils
import matplotlib.pyplot as plt
#from qutip import*

Data = []
Ave = []

N = 11
W = 10.0
t = 1.0
G = 10.0
# seed=1

save = True
path0 = 'results/center/'


for i in range(N):
    Sz_total = []
    Sp_total = []
    for seed in range(1, 21):
        name = 'N='+str(N)+' W=' + str(W)+' t='+str(t) + \
            ' g=' + str(G) + ' seed=' + str(seed)
        path = path0 + name + '/store.p'
        if os.path.exists(path):
            data = utils.load_vars(path)

            eps = data['eps']
            y1 = data['y1']
            y2 = data['y2']
            index = data['index']
            t1 = data['t1']
            t2 = data['t2']
            gamma=data['gamma']
    #        S_z=[eps[i]*y1[i] for i in index['z']]
            t_total = np.append(t1, t2)
            Sz_total.append(np.append(y1[index['z'][i]], y2[index['z'][i]]))
            Sp_total.append(np.append(y1[index['+'][i]], y2[index['+'][i]]))

    Sz_ave = (np.mean(Sz_total, 0))
    Sp_ave = (np.mean(Sp_total, 0))
    plt.figure()
    plt.plot(t_total*t, Sp_ave, label="$Re <S^+_{%d} $" % (i))    
    plt.plot(t_total*t, Sz_ave, label="$Re <S^z_{%d} $" % (i))

    plt.legend()
    plt.ylabel("site {}".format(i))
    plt.ylim(-0.5, 0.5)
    plt.axhline(y=0, color='grey', linestyle='--')
    plt.xlabel('t * $\overline{J}$')
    plt.suptitle(
        'XY model, N=%d, W=%d,  $\overline{J}$=%.2f  $\gamma$=%.1f' % (N, W, t, gamma[i]))
    if save:
        savename =path0+ 'Ave '+'N='+str(N)+' W=' + str(W)+' t='+str(t) +' g=' + str(G)
        os.makedirs(savename, exist_ok=True)
        plt.savefig(savename +"/site {}.png".format(i))
