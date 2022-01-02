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

#eps=0*eps+1


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

times=np.append(np.linspace(0, 100,1000), np.linspace(100, 500,1000))
z=np.append(z1,z2)
plt.plot(times,z)
plt.axvline(x=100, color='grey', linestyle='--')
plt.title('$\sum_i <S_i^z>$')
#plt.title('$\sum_i \epsilon_i<S_i^z>$')        
#plt.plot(z1)
#plt.plot(z2)        

"""
check level spacing
"""


s1=[]
for t in range(1000):
    l1=[]
    for i in range(N):
        l1.append(E1[i][t])
    l1.sort()
    s1.append(np.diff(l1))
    
s1=np.ndarray.flatten(np.array(s1))

plt.figure()
plt.hist(s1, bins=100)







