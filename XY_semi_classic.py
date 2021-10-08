# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 18:38:51 2021

@author: westl
"""

import numpy as np
from XY_class import*
from XY_ode_funs import ode_funs
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time 
from datetime import datetime 
from tqdm import tqdm
from qutip import*


L=2
N=6

model=XY(L,N)

H=model.get_Hamiltonian()
rho0=model.generate_random_density()

gamma=model.generate_gamma()

eps=model.eps
J=model.J
U=model.U

y0=expect(model.Sz+model.Sp, rho0)
   
ode_funs=ode_funs(N, eps, J, U, gamma, Diss='dephasing')

fun=ode_funs.fun_1st

index=ode_funs.flat_index(single_ops=['z','+'], double_ops=[], index={}) 

t_0=0
t_1=10
t_span=(t_0,t_1)
t_eval=np.linspace(t_0, t_1, 1000) 


result1=solve_ivp(fun, t_span=t_span, y0=y0, t_eval=t_eval, args=[index])   

plt.plot(result1.t, result1.y[0], label='Semiclassical approx')  