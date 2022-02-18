# -*- coding: utf-8 -*-
"""
Created on Fri Oct  8 16:54:57 2021

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
import random
#from Lindblad_solver import Lindblad_solve
from energy_paras import Energycomputer, Jcomputer, Ucomputer, Gammacomputer



L=2
N=5


show_type='z'
#show_ind=random.randrange(N)
show_ind=(0)

model=XY(L,N)

H=model.get_Hamiltonian(W=1, t=0.1, u=0)
#states,rho0=model.generate_coherent_density()
#rho0=model.generate_random_density(seed=1)
up_sites=[i for i in range(0,N,2)]
states, rho0=model.generate_up(up_sites)#print(rho0)
Sz=model.Sz
Sp=model.Sp
Sm=model.Sm
SpSm=model.generate_SpSm()
#G=0.0000000000000001
G=1

#gamma=model.generate_gamma(G) # if gamma is too large will cause too stiff ode, thus need to increase number of steps correspondingly.
gamma=Gammacomputer(N).central_g(G)

eps=model.eps
J=model.J
U=model.U

#Diss='dephasing' 
Diss='dissipation'

"""
sloving by semi-classical 1st order: <S1*S2>=<S1>*<S2>
"""


y1=expect(Sz+Sp, rho0)
   


ode_funs=ode_funs(N, eps, J, U, gamma, Diss=Diss) # chose the jump opperator for 'dephasing' or 'dissipation'

fun=ode_funs.fun_1st

index=ode_funs.flat_index(single_ops=['z','+'], double_ops=[], index={}) 

t_0=0
t_1=100
t_span=(t_0,t_1)
t_eval=np.linspace(t_0, t_1, 1000 )

with tqdm(total=100, unit="‰") as pbar:

    result1=solve_ivp(fun, t_span=t_span, y0=y1, t_eval=t_eval, args=[index, pbar, [t_0, (t_1-t_0)/100]])  

plt.plot(result1.t, result1.y[index[show_type][show_ind]], label='1st-order approx')



"""
sloving by semi-classical 2nd order: <S^+S^-S^z>=<S^+S^-><S^z>
"""

y2=expect(Sz+SpSm, rho0)
   
fun=ode_funs.fun_2nd

index=ode_funs.flat_index(single_ops=['z'], double_ops=['+-'], index={}) 



result2=solve_ivp(fun, t_span=t_span, y0=y2, t_eval=t_eval, args=[index])  

plt.plot(result2.t, result2.y[index[show_type][show_ind]], label='2nd-order approx')




"""
sloving by Qutip Lindblad

"""
if Diss=='dephasing':
    diss=Sz
else:
    diss=Sm
    

e_ops=Sz+SpSm # list of expectation values to evaluate

c_ops=[]

for i in range(N):
    c_ops.append(np.sqrt(gamma[i])*diss[i])


t_span=(t_0,t_1)
times=t_eval

ops=Options(tidy=(False), average_expect=(False))

    
result3=mesolve(H, rho0, times, c_ops, e_ops, progress_bar=True, options=None) 

plt.plot(result3.times, result3.expect[index[show_type][show_ind]], label='Qutip solved Lindblad') 






plt.title('XY model L=%d, N=%d  gamma=%dW for ' % (L,N,G)+Diss)
plt.xlabel('t')
plt.ylabel("$<S^{}_{}>$".format(show_type, show_ind))
plt.ylim(-0.6, 0.6)
plt.legend()  
