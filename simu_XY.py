# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:45:27 2021

@author: westl
"""

import numpy as np
from XY_class import XY
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time 
from datetime import datetime 
from tqdm import tqdm
from qutip import*

L=2
N=3
gamma=1

pick=0

model=XY(L,N)

H=model.get_Hamiltonian()
rho0=model.generate_random_density()

Sz=model.Sz
Sp=model.Sp
Sm=model.Sm

diss=Sz

e_ops=Sz+Sp+Sm # list of expectation values to evaluate

c_ops=[]

for i in range(N):
    c_ops.append(np.square(gamma)*diss[i])


t_0=0
t_1=10
t_span=(t_0,t_1)
times=np.linspace(t_0, t_1, 100) 
    
result=mesolve(H, rho0, times, c_ops, e_ops, progress_bar=True) 

plt.plot(result.times, result.expect[pick], label='numerical solved Lindblad') 
plt.title('XY model L=%d, N=%d ' % (L,N))
plt.xlabel('t')
plt.ylabel('$<S^z_{%d}>$' % pick)
plt.legend()  




    
    
 