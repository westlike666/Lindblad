# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 18:45:27 2021

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
from Lindblad_solver import Lindblad_solve



L=2
N=6


show_type='+'
#show_ind=random.randrange(N)
show_ind=0

model=XY(L,N)

H=model.get_Hamiltonian(W=1, t=0.1, u=0)
#rho0=model.generate_coherent_density()
rho0=model.generate_random_density(seed=2)
#print(rho0)
Sz=model.Sz
Sp=model.Sp
Sm=model.Sm
#G=0.0000000000000001
G=1

gamma=model.generate_gamma(G) # if gamma is too large will cause too stiff ode, thus need to increase number of steps correspondingly.


"""
sloving by semi-classical 1st order: <S1*S2>=<S1>*<S2>
"""
eps=model.eps
J=model.J
U=model.U

y0=expect(model.Sz+model.Sp, rho0)
   
#Diss='dephasing' 
Diss='dissipation'

ode_funs=ode_funs(N, eps, J, U, gamma, Diss=Diss) # chose the jump opperator for 'dephasing' or 'dissipation'

fun=ode_funs.fun_1st

index=ode_funs.flat_index(single_ops=['z','+'], double_ops=[], index={}) 

t_0=0
t_1=20
t_span=(t_0,t_1)
t_eval=np.linspace(t_0, t_1, 1000 )


result1=solve_ivp(fun, t_span=t_span, y0=y0, t_eval=t_eval, args=[index])  

plt.plot(result1.t, result1.y[index[show_type][show_ind]], label='1st-order approx')




"""
sloving by Qutip Lindblad

"""
if Diss=='dephasing':
    diss=Sz
else:
    diss=Sm
    

e_ops=Sz+Sp # list of expectation values to evaluate

c_ops=[]

for i in range(N):
    c_ops.append(np.sqrt(gamma[i])*diss[i])


t_span=(t_0,t_1)
times=t_eval

ops=Options(tidy=(False), average_expect=(False))

    
result2=mesolve(H, rho0, times, c_ops, e_ops, progress_bar=True, options=None) 

plt.plot(result2.times, result2.expect[index[show_type][show_ind]], label='Qutip solved Lindblad') 


result3, expect_value=Lindblad_solve(H, rho0, t_span, t_eval, c_ops=c_ops, e_ops=e_ops)  

plt.plot(result3.t, expect_value[index[show_type][show_ind]], label='solve_ivp solved Lindblad') 



plt.title('XY model L=%d, N=%d  gamma=%dW for ' % (L,N,G)+Diss)
plt.xlabel('t')
plt.ylabel("$<S^{}_{}>$".format(show_type, show_ind))
plt.legend()  






    
    
 