# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 16:08:55 2021

@author: User
"""

import utils
import matplotlib.pyplot as plt
import numpy as np


path='results/2021-12-6_15_39_39'
data=utils.load_vars(path+'/store.p')
save=True



y1=data['y1']
y2=data['y2']
index=data['index']
N=data['N']
eps=data['eps']
t=0.15
L=2
G=1

t1=np.linspace(0, 100, 1000)
t2=np.linspace(100,500,1000)
t_total=np.append(t1,t2)



def plot_evolution(show_type='z', show_ind=0):

#    t_total=np.append(result1.times, result2.times)
    y_total=np.append(y1[index[show_type][show_ind]].real,y2[index[show_type][show_ind]].real)
    plt.figure(show_ind)
    #plt.subplot(211)
    plt.plot(t_total, y_total, label="$Re <S^{}_{}>$".format(show_type, show_ind))
    plt.ylabel("site {}".format(show_ind))
    plt.axhline(y=-0.5, color='grey', linestyle='--')
    plt.legend()
    # plt.subplot(212)
    # plt.plot(t_total, result1.y[index[show_type][show_ind]].imag, label='1st-order approx') 
    # plt.plot(t_total, result2.expect[index[show_type][show_ind]].imag, label='Qutip solved Lindblad')
    # plt.ylabel("$Im <S^{}_{}>$".format(show_type, show_ind))
    # plt.legend()
    plt.xlabel('t')
    plt.suptitle('XY model L=%d, N=%d  eps=%.2f  t=%.2f W  g=%.1f W' % (L,N, eps[show_ind],t,G))
    if save:
        plt.savefig(path+"/site {}.png".format(show_ind))    
for show_ind in range(N):
    plot_evolution('+',show_ind)
    plot_evolution('z', show_ind)
