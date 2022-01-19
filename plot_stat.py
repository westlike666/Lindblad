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

#path='results/2021-11-11_18_18_21/'
#path='results/2021-11-12_16_26_43/'
#path='results/2021-11-15_02_19_06/'
#path='results/2021-11-15_15_44_38/'
#path='results/2021-11-16_13_38_09/'
#path='results/2021-11-17_18_50_05/'
#path='results/2021-11-26_17_38_49/'
#path='results/2021-12-2_15_07_58/'
#path='results/2021-12-6_15_39_39/'
#path='results/2022-1-1_19_48_53/'
path='results/2022-1-11_17_51_04/'
#path='results/2022-1-13_20_20_11/'
#path='results/2022-1-15_21_13_50/'

data=utils.load_vars(path+'store.p')

N=data['N']
eps=data['eps']
y1=data['y1']
y2=data['y2']
index=data['index']

"""
check conservation
"""

energy=False

if not energy:
    eps=0*eps+1


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

t1=np.linspace(0, 100,1000)
t2=np.linspace(100, 500,1000)

#z=np.append(z1,z2)
plt.plot(t1,z1)
plt.plot(t2,z2)
plt.xlabel('t')
plt.axvline(x=100, color='grey', linestyle='--')
if not energy:
    plt.title('$\sum_i <S_i^z>$')
    plt.savefig(path+'decay_spin.png')
else:    
    plt.title('$\sum_i \epsilon_i<S_i^z>$')
    plt.savefig(path+'decay_energy.png')


        
#plt.plot(z1)
#plt.plot(z2)        

"""
check level spacing
"""


s1=[]
s2=[]
for t in range(1000):
    l1=[]
    l2=[]
    for i in range(N):
        l1.append(E1[i][t])
        l2.append(E2[i][t])
    l1.sort()
    l2.sort()    
    s1.append(np.diff(l1))
    s2.append(np.diff(l2))    
    
s1=np.ndarray.flatten(np.array(s1))
s2=np.ndarray.flatten(np.array(s2))
#s1=s1/s1.mean()
#s2=s2/s2.mean()


plt.figure()
plt.hist(s1, bins=500,alpha=1,density=True)
plt.hist(s2, bins=500,alpha=0,density=True)


if not energy:
    plt.title('level spacing of $ <S_i^z>$')
else:    
    plt.title('level spacing of $\epsilon_i<S_i^z>$')

#plt.ylim((0,2500))

plt.figure()
plt.hist(s1, bins=500,alpha=0,density=True)
plt.hist(s2, bins=500,alpha=1,density=True)





