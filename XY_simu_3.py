# -*- coding: utf-8 -*-
"""
Created on Fri Oct  12 16:54:57 2021

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



L=2
N=2


show_type='+-'
#show_ind=random.randrange(N)
show_ind=(0,1)

model=XY(L,N)
H=model.get_Hamiltonian(W=1, t=0.1, u=0.00, seed=1)
print(model.eps, model.J, model.U)
#rho0=model.generate_coherent_density()
rho0=model.generate_random_density(seed=1)
#print(rho0)
Sz=model.Sz
Sp=model.Sp
Sm=model.Sm
SpSm=model.generate_SpSm()
SpSp=model.generate_SpSp()
SmSm=model.generate_SmSm()
SpSz=model.generate_SpSz()
SmSz=model.generate_SmSz()
SzSz=model.generate_SzSz()


G=1
#G=0

gamma=model.generate_gamma(G) # if gamma is too large will cause too stiff ode, thus need to increase number of steps correspondingly.

eps=model.eps
J=model.J
U=model.U

#Diss='dephasing' 
Diss='dissipation'

"""
sloving by semi-classical 1st order: <S1*S2>=<S1>*<S2>
"""

e_ops=Sz+Sp

y1=expect(e_ops, rho0)
   


ode_funs=ode_funs(N, eps, J, U, gamma, Diss=Diss) # chose the jump opperator for 'dephasing' or 'dissipation'

fun=ode_funs.fun_1st

index=ode_funs.flat_index(single_ops=['z', '+'], double_ops=[], index={}) 

t_0=0
t_1=100
t_span=(t_0,t_1)
t_eval=np.linspace(t_0, t_1, 1000)

with tqdm(total=100, unit="‰") as pbar:
    result1=solve_ivp(fun, t_span=t_span, y0=y1, t_eval=t_eval, args=[index, pbar, [t_0, (t_1-t_0)/100]])  





"""
sloving by semi-classical 2nd order: <abc>=<ab><c>+<ac><b>+<bc><a>-2<a><b><c>  
"""
e_ops=Sz+Sp+Sm+SpSm+SpSp+SmSm+SpSz+SmSz+SzSz

y2=expect(e_ops, rho0)
   
fun=ode_funs.fun_2nd_all

index=ode_funs.flat_index(['z', '+', '-'],[ '+-', '++','--','+z', '-z', 'zz' ], index={}) 




result2=solve_ivp(fun, t_span=t_span, y0=y2, t_eval=t_eval, args= [index]) 






"""
sloving by Qutip Lindblad

"""
if Diss=='dephasing':
    diss=Sz
else:
    diss=Sm
    

e_ops=e_ops # list of expectation values to evaluate

c_ops=[]

for i in range(N):
    c_ops.append(np.sqrt(gamma[i])*diss[i])


t_span=(t_0,t_1)
times=t_eval

ops=Options(tidy=(False), average_expect=(False))

    
result3=mesolve(H, rho0, times, c_ops, e_ops, progress_bar=True, options=None) 


#plt.plot(result1.t, result1.y[index[show_type][show_ind]], label='1st-order approx')
plt.plot(result2.t, result2.y[index[show_type][show_ind]], label='2nd-order approx')
plt.plot(result3.times, result3.expect[index[show_type][show_ind]], label='Qutip solved Lindblad') 
plt.title('XY model L=%d, N=%d  gamma=%dW for ' % (L,N,G)+Diss)
plt.xlabel('t')
plt.ylabel("$<S^{}_{}>$".format(show_type, show_ind))
plt.legend()  

plt.figure()
#plt.plot(result1.t, result1.y[index[show_type][show_ind]].imag, label='1st-order approx')
plt.plot(result2.t, result2.y[index[show_type][show_ind]].imag, label='2nd-order approx')
plt.plot(result3.times, result3.expect[index[show_type][show_ind]].imag, label='Qutip solved Lindblad') 
plt.title('XY model L=%d, N=%d  gamma=%dW for ' % (L,N,G)+Diss)
plt.xlabel('t')
plt.ylabel("$<S^{}_{}>^*$".format(show_type, show_ind))
plt.legend()