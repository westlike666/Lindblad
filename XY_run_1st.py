# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:00:17 2021

@author: User
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
from energy_paras import Energycomputer, Jcomputer, Ucomputer, Gammacomputer


L=2
N=3

W=1
t=1
u=0
G1=0.0
G2=0.1

t_0=0
t_1=100
steps=1000

t_2=200


seed=1

show_type='z' 
#show_ind=random.randrange(N)
show_ind=(1)

model=XY(L,N)

eps_comp=Energycomputer(N,seed).uniformrandom_e(W)
J_comp=Jcomputer(N, nn_only=False, scaled=False, seed=seed).uniformrandom_j(t)
U_comp=Ucomputer(N, nn_only=False, scaled=True, seed=seed).uniformrandom_u(u)
#G_comp=Gammacomputer(N).constant_g(G)    

H=model.get_Hamiltonian2(eps_comp, J_comp, U_comp)
gamma=model.generate_gamma(G1)

#print(model.eps, model.J, model.U)
#rho0=model.generate_coherent_density()
rho0=model.generate_random_density()
#print(rho0)
Sz=model.Sz
Sp=model.Sp
Sm=model.Sm


 # if gamma is too large will cause too stiff ode, thus need to increase number of steps correspondingly.

eps=model.eps
J=model.J
U=model.U

#Diss='dephasing' 
Diss='dissipation'

"""
sloving by semi-classical 1st order: <S1*S2>=<S1>*<S2> for the first stage with decay G1
"""

e_ops=Sz+Sp

y1=expect(e_ops, rho0)
   


ode_class=ode_funs(N, eps, J, U, gamma, Diss=Diss) # chose the jump opperator for 'dephasing' or 'dissipation'

fun=ode_class.fun_1st

index=ode_class.flat_index(single_ops=['z', '+'], double_ops=[], index={}) 


t_span=(t_0,t_1)
t_eval=np.linspace(t_0, t_1, steps)

with tqdm(total=100, unit="‰") as pbar:
    result1=solve_ivp(fun, t_span=t_span, y0=y1, t_eval=t_eval, args=[index, pbar, [t_0, (t_1-t_0)/100]])  

plt.figure()
plt.plot(result1.t, result1.y[index[show_type][show_ind]], label='Re')    
plt.plot(result1.t, result1.y[index[show_type][show_ind]].imag, label='Im')    
plt.title('XY model L=%d, N=%d  t=%.1f W  gamma=%.1f W for ' % (L,N,t,G1)+Diss)
plt.xlabel('t')
plt.ylabel("$<S^{}_{}>$".format(show_type, show_ind))
plt.legend()   

    
"""
sloving by semi-classical 1st order: <S1*S2>=<S1>*<S2> for the second stage with decay G2
"""

y2=result1.y[:,-1]



gamma=model.generate_gamma(G2)
ode_class=ode_funs(N, eps, J, U, gamma, Diss=Diss)
fun=ode_class.fun_1st

t_span=(t_1, t_2)
t_eval=np.linspace(t_1, t_2, steps)

with tqdm(total=100, unit="‰") as pbar:
    result2=solve_ivp(fun, t_span=t_span, y0=y2, t_eval=t_eval, args=[index, pbar, [t_1, (t_2-t_1)/100]]) 


"""
plotting the total evolution
"""


t_total=np.append(result1.t,result2.t)
y_total=np.append(result1.y[index[show_type][show_ind]],result2.y[index[show_type][show_ind]])

plt.figure()
plt.plot(t_total, y_total, label='Re')    
plt.plot(t_total, y_total.imag, label='Im')    
plt.title('XY model L=%d, N=%d  t=%.1f W  gamma=%.1f W for ' % (L,N,t,G2)+Diss)
plt.xlabel('t')
plt.ylabel("$<S^{}_{}>$".format(show_type, show_ind))
plt.legend()  


plt.figure()
plt.plot(y1,'o', label='y1')
plt.plot(y2, 'x', label='y2')
plt.legend()
    
 
    