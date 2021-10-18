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
from energy_paras import Energycomputer, Jcomputer, Ucomputer, Gammacomputer


L=2
N=3

W=1
t=0.1
u=0

seed=1

show_type='z'
#show_ind=random.randrange(N)
show_ind=0

model=XY(L,N)


eps=Energycomputer(N,seed).uniformrandom_e(W)
J=Jcomputer(N, nn_only=False, scaled=False, seed=seed).uniformrandom_j(t)
U=Ucomputer(N, nn_only=False, scaled=True, seed=seed).uniformrandom_u(u)

#G_comp=Gammacomputer(N).constant_g(G)    

H=model.get_Hamiltonian2(eps, J, U)

rho0=model.generate_coherent_density(alpha=1*np.pi/2.5)
#rho0=model.generate_random_density(seed=1)

#print(rho0)
Sz=model.Sz
Sp=model.Sp
Sm=model.Sm
#G=0.000000000000000
G=0.1

gamma=model.generate_gamma(G) # if gamma is too large will cause too stiff ode, thus need to increase number of steps correspondingly.


"""
sloving by semi-classical 1st order: <S1*S2>=<S1>*<S2>
"""

y0=expect(model.Sz+model.Sp, rho0)
   
#Diss='dephasing' 
Diss='dissipation'

ode_funs=ode_funs(N, eps, J, U, gamma, Diss=Diss) # chose the jump opperator for 'dephasing' or 'dissipation'

fun=ode_funs.fun_1st

index=ode_funs.flat_index(single_ops=['z','+'], double_ops=[], index={}) 

t_0=0
t_1=100
t_span=(t_0,t_1)
t_eval=np.linspace(t_0, t_1, 1000 )


with tqdm(total=100, unit="‰") as pbar:
    result1=solve_ivp(fun, t_span=t_span, y0=y0, t_eval=t_eval, args=[index,pbar, [t_0, (t_1-t_0)/100]])  






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



# result3, expect_value=Lindblad_solve(H, rho0, t_span, t_eval, c_ops=c_ops, e_ops=e_ops)  

# plt.plot(result3.t, expect_value[index[show_type][show_ind]], label='solve_ivp solved Lindblad') 

plt.figure()
plt.plot(result1.t, result1.y[index[show_type][show_ind]].real, label='1st-order approx')
plt.plot(result2.times, result2.expect[index[show_type][show_ind]].real, label='Qutip solved Lindblad') 
plt.title('XY model L=%d, N=%d  gamma=%dW for ' % (L,N,G)+Diss)
plt.xlabel('t')
plt.ylabel("$<S^{}_{}>$".format(show_type, show_ind))
plt.legend()  

plt.figure()
plt.plot(result1.t, result1.y[index[show_type][show_ind]].imag, label='1st-order approx')
plt.plot(result2.times, result2.expect[index[show_type][show_ind]].imag, label='Qutip solved Lindblad') 
plt.title('XY model L=%d, N=%d  gamma=%dW for ' % (L,N,G)+Diss)
plt.xlabel('t')
plt.ylabel("$<S^{}_{}>^*$".format(show_type, show_ind))
plt.legend()  





    
    
 