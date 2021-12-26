# -*- coding: utf-8 -*-
"""
Created on Sat Dec 25 19:11:41 2021

@author: westl
"""

import numpy as np
import os
import utils
import matplotlib.pyplot as plt
#from qutip import*
import random
from numpy.random import default_rng

path='results/2021-11-11_18_18_21/store.p'

data=utils.load_vars(path)

N=data['N']
eps=data['eps']
y1=data['y1']
y2=data['y2']
index=data['index']

"""
check conservation
"""

E1=[eps[i]*y1[i] for i in index['z']]
E2=[eps[i]*y2[i] for i in index['z']]

# E1=[y1[i] for i in index['z']]
# E2=[y2[i] for i in index['z']]

t=0

z1=[]
z2=[]

for t in range(1000):
    z1.append(0j)
    z2.append(0j)
    for i in range(N):
        z1[t] += E1[i][t]
        z2[t] += E2[i][t]
        
plt.plot(z1)
plt.plot(z2)        

"""
check level spacing
"""

l1=[eps[i]*y1[i] for i in index['z']]
l2=[eps[i]*y2[i] for i in index['z']]












