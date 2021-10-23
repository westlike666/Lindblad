# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:53:05 2021

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
N=7

W=1
t=1
u=0

G=1

seed=None

show_type='z'
#show_ind=random.randrange(N)
show_ind=0

model=XY(L,N)


eps=Energycomputer(N,seed).uniformrandom_e(W)
J=Jcomputer(N, nn_only=False, scaled=True, seed=seed).constant_j(t)
U=Ucomputer(N, nn_only=False, scaled=True, seed=seed).uniformrandom_u(u)
gamma=Gammacomputer(N).central_g(G)
#gamma=Gammacomputer(N).boundary_g(G)
#gamma=Gammacomputer(N).site_g(G,[2,3,4])

H=model.get_Hamiltonian2(eps, J, U)

#rho0=model.generate_coherent_density(alpha=1*np.pi/9)
rho0=model.generate_random_density(pure=True,seed=10)

#print(rho0)
Sz=model.Sz
Sp=model.Sp
Sm=model.Sm





"""
sloving by semi-classical 1st order: <S1*S2>=<S1>*<S2>
"""

   
#Diss='dephasing' 
Diss='dissipation'

ode_funs=ode_funs(N, eps, J, U, gamma, Diss=Diss) # chose the jump opperator for 'dephasing' or 'dissipation'


index=ode_funs.flat_index(single_ops=['z','+'], double_ops=[], index={}) 

t_0=0
t_1=100
t_span=(t_0,t_1)
t_eval=np.linspace(t_0, t_1, 1000)



"""
sloving by Qutip Lindblad

"""

e_ops=Sz+Sp # list of expectation values to evaluate

times1=t_eval

    
result1=mesolve(H, rho0, times1, progress_bar=True) 
y1=expect(e_ops, result1.states)

rho1=result1.states[-1]

if Diss=='dephasing':
    diss=Sz
else:
    diss=Sm
    
c_ops=[]

for i in range(N):
    c_ops.append(np.sqrt(gamma[i])*diss[i])
    
t_2=500
t_span=(t_1, t_2)
times2=np.linspace(t_1, t_2, 1000)    

result2=mesolve(H, rho1, times2, c_ops, progress_bar=True)
y2=expect(e_ops, result2.states) 


# result3, expect_value=Lindblad_solve(H, rho0, t_span, t_eval, c_ops=c_ops, e_ops=e_ops)  

# plt.plot(result3.t, expect_value[index[show_type][show_ind]], label='solve_ivp solved Lindblad') 



def plot_evolution(show_type='z', show_ind=0):
    t_total=np.append(result1.times, result2.times)
    y_total=np.append(y1[index[show_type][show_ind]],y2[index[show_type][show_ind]])

    plt.figure(show_ind)
    #plt.subplot(211)
    plt.plot(t_total, y_total, label="$Re <S^{}_{}>$".format(show_type, show_ind))
    plt.ylabel("site {}".format(show_ind))
    plt.axhline(y=-0.5, color='grey', linestyle='--')
    plt.legend() 
    # plt.subplot(212)
    # plt.plot(t_total, result1.y[index[show_type][show_ind]].imag, label='1st-order approx') 
    # plt.plot(t_total, result2.expect[index[show_type][show_ind]].imag, label='Qutip solved Lindblad')
    # plt.ylabel("$Im <S^{}_{}>$".format(show_type, show_ind))
    # plt.legend()
    plt.xlabel('t')
    plt.suptitle('XY model L=%d, N=%d  t=%.1f W  g=%.1f W from one end' % (L,N,t,G))

for show_ind in range(N):
    plot_evolution('+',show_ind)
    plot_evolution('z', show_ind)