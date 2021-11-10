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
from numpy.random import default_rng
import utils 
import os


L=2
N=5

W=1
t=0.15
u=0

G=1
seed=None

show_type='z'
#show_ind=random.randrange(N)
show_ind=0

save=True

if save:
    path='results/'+utils.get_run_time()
    os.mkdir(path)


model=XY(L,N)


eps=Energycomputer(N,seed).uniformrandom_e()
J=Jcomputer(N, nn_only=False, scaled=True, seed=seed).constant_j(t)
U=Ucomputer(N, nn_only=False, scaled=True, seed=seed).uniformrandom_u(u)
gamma=Gammacomputer(N).central_g(G)
#gamma=Gammacomputer(N).boundary_g(G)
#gamma=Gammacomputer(N).site_g(G,[2,3,4])
#gamma=Gammacomputer(N).constant_g(G)

H=model.get_Hamiltonian2(eps, J, U)
#states,rho0=model.generate_coherent_density(alpha=1*np.pi/2.1)
#states, rho0=model.generate_random_density(pure=True, seed=None) # 5, 6, 10, 11

rng=default_rng(seed=1)
up_sites=rng.choice(N, N//2, replace=False)
states, rho0=model.generate_up(up_sites)

print(up_sites)
Sz=model.Sz
Sp=model.Sp
Sm=model.Sm






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
t_1=500
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

    
#result2=mesolve(H, rho0, times, c_ops, e_ops, progress_bar=True, options=None) 



result3, expect_value=Lindblad_solve(H, rho0, t_span, t_eval, c_ops=c_ops, e_ops=e_ops)  

# plt.plot(result3.t, expect_value[index[show_type][show_ind]], label='solve_ivp solved Lindblad') 

# z1=result1.y[index[show_type][show_ind]]
# z2=result2.expect[index[show_type][show_ind]]

# err=np.linalg.norm(z1-z2)/np.linalg.norm(z2)

# print("--- error for N={} sites is: {} ---\n" .format(N, err))


def plot_evolution(show_type='z', show_ind=0):
    
    time=result1.t
    plt.figure()
    #plt.subplot(211)
    plt.plot(time, result1.y[index[show_type][show_ind]].real, label='1st-order approx') 
    #plt.plot(time, result2.expect[index[show_type][show_ind]], label='Qutip solved Lindblad')
    plt.plot(result3.t, np.array(expect_value[index[show_type][show_ind]]).real ,label='solve_ivp solved Lindblad') 
    plt.ylabel("$Re <S^{}_{}>$".format(show_type, show_ind))
    plt.axhline(y=-0.5, color='grey', linestyle='--')
    plt.legend() 
    # plt.subplot(212)
    # plt.plot(time, result1.y[index[show_type][show_ind]].imag, label='1st-order approx') 
    # plt.plot(time, result2.expect[index[show_type][show_ind]].imag, label='Qutip solved Lindblad')
    # plt.ylabel("$Im <S^{}_{}>$".format(show_type, show_ind))
    # plt.legend()
    plt.xlabel('t')
    plt.suptitle('XY model L=%d, N=%d  t=%.1f W  g=%.1f W for ' % (L,N,t,G)+Diss)
    if save:
        plt.savefig(path+"/site {}".format(show_ind)+show_type+".pdf")
        
for show_ind in range(N):
    plot_evolution('+',show_ind)

    plot_evolution('z', show_ind)


# y1=y0
# y2=result1.y[:,-1]

# plt.figure()
# plt.plot(y1,'o', label='initial')
# plt.plot(y2, 'x', label='final')
# plt.legend()
    


    
    
 