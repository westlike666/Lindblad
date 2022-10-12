# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 15:01:20 2022

@author: westl
"""

import scipy
import numpy as np
import os
import utils
from XY_class import*
from MPC_ode_funs import ode_funs
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time 
from datetime import datetime 
from tqdm import tqdm
from qutip import*
import random
from numpy.random import default_rng
#from Lindblad_solver import Lindblad_solve
from energy_paras import Energycomputer, Jcomputer, Ucomputer, Gammacomputer

save=False

if save:
    path='results/'+utils.get_run_time()
    os.mkdir(path)

L=2
N=5
W=0
t=1
u=0

G=0

seed=None
single_type='z'
double_type='+-'
#show_ind=random.randrange(N)
show_ind=0

model=XY(L,N)


eps=Energycomputer(N,seed).uniformrandom_e(W)
J=Jcomputer(N, nn_only=False, scaled=True, seed=seed).constant_j(t)
U=Ucomputer(N, nn_only=False, scaled=True, seed=seed).uniformrandom_u(u)
gamma=Gammacomputer(N).central_g(G)
#gamma=Gammacomputer(N).boundary_g(G)
#gamma=Gammacomputer(N).site_g(G,[0,6])
#gamma=Gammacomputer(N).constant_g(G)

H=model.get_Hamiltonian_MPC(eps, J)

#states,rho0=model.generate_random_density(seed=None, pure=True) #seed works for mixed state
states,rho0=model.generate_coherent_density(alpha=np.pi/4) 
#states,rho0=model.generate_random_ket()
#rng=default_rng(seed=1)
#up_sites=rng.choice(N, N//2, replace=False)
up_sites=[i for i in range(0,N,2)]
#states, rho0=model.generate_up(up_sites)

#print(states)
print(up_sites)
print(eps)

#print(rho0)
Sz=model.Sz
Sp=model.Sp
Sm=model.Sm


   
#Diss='dephasing' 
Diss='dissipation'

# ode_funs=ode_funs(N, eps, J, U, gamma, Diss=Diss) # chose the jump opperator for 'dephasing' or 'dissipation'
# index=ode_funs.flat_index(single_ops=['z','+'], double_ops=[], index={}) 

t_0=0
t_1=100
t_span1=(t_0,t_1)
times1=np.linspace(t_0, t_1, 100)

t_2=500
t_span2=(t_1, t_2)
times2=np.linspace(t_1, t_2, 100)



s=['z', '+', '-']  # keep the same order as the 1st-order does 
ss=['+-', '++', '--', '+z', '-z', 'zz']
e_ops=model.generate_single_ops(s)+model.generate_double_ops(ss)





#%%
"""
sloving by Qutip Lindblad

"""
# phase1


result1_exact=mesolve(H, rho0, times1, progress_bar=True) 
y1_exact=expect(e_ops, result1_exact.states)


#phase2

rho1=result1_exact.states[-1]

if Diss=='dephasing':
    diss=Sz
else:
    diss=Sm
    
c_ops=[]

# for i in range(N):
#     c_ops.append(np.sqrt(gamma[i])*diss[i])
    


for i in range(N):
    c_ops.append(np.sqrt(gamma[i])*diss[i]) # constant c_ops

#    c_ops.append([np.sqrt(gamma[i])*diss[i],np.sin(times2-t_1)/10]) # periodic c_ops
 
    

result2_exact=mesolve(H, rho1, times2, c_ops, args={},progress_bar=True)
y2_exact=expect(e_ops, result2_exact.states) 




#%%

"""
sloving by 2nd order mean-filed: <abc>=<ab><c>+<ac><b>+<bc><a>-2<a><b><c>  

"""

#phase1





ode_class=ode_funs(N, eps, J, U, gamma=0*gamma, Diss=Diss) 
index=ode_class.flat_index(s, ss, index={})

fun=ode_class.fun_2nd_new
y0_MPC=expect(e_ops, rho0)



result1_MPC=solve_ivp(fun, t_span=t_span1, t_eval=times1, y0=y0_MPC, args=[index]) # no progressing bar yet
t1_MPC=result1_MPC['t']
y1_MPC=result1_MPC['y']

#phase2

ode_class=ode_funs(N, eps, J, U, gamma=gamma, Diss=Diss)
fun=ode_class.fun_2nd_new

result2_MPC=solve_ivp(fun, t_span=t_span2, t_eval=times2, y0=y1_MPC[:,-1], args=[index]) # no progressing bar yet
t2_MPC=result2_MPC['t']
y2_MPC=result2_MPC['y']

#%%

"""
plotting
"""




def plot_evolution(show_type='z', show_ind=0):

    t_total=np.append(result1_exact.times, result2_exact.times)
    y_total=np.append(y1_exact[index[show_type][show_ind]].real,y2_exact[index[show_type][show_ind]].real)   
   # y_total_semi=np.append(y1_semi[index[show_type][show_ind]].real,y2_semi[index[show_type][show_ind]].real)
    y_total_MPC=np.append(y1_MPC[index[show_type][show_ind]],y2_MPC[index[show_type][show_ind]])
    
    plt.figure()
    #plt.subplot(211)
    plt.plot(t_total, y_total, label="Qutip exact $Re <S^{}_{}>$".format(show_type, show_ind))
   # plt.plot(np.append(t1_semi,t2_semi), y_total_semi, label="1st order $Re <S^{}_{}>$".format(show_type, show_ind))
    plt.plot(np.append(t1_MPC,t2_MPC), y_total_MPC, label="2nd order $Re <S^{}_{}>$".format(show_type, show_ind))
    plt.ylabel("site {}".format(show_ind))
    #plt.ylim(-0.6,0.6)
    plt.axhline(y=-0.5, color='grey', linestyle='--')
    plt.legend()
    plt.xlabel('t')
    plt.suptitle('XY model L=%d, N=%d  eps=%.2f  t=%.2f W  g=%.1f W' % (L,N, eps[show_ind],t,G))
    if save:
        plt.savefig(path+"/site {}.png".format(show_ind))    
        
def plot_double_evolution(show_type, ind1,ind2):

    t_total=np.append(result1_exact.times, result2_exact.times)
    y_total=np.append(y1_exact[index[show_type][ind1,ind2]].real,y2_exact[index[show_type][ind1,ind2]].real)   
   # y_total_semi=np.append(y1_semi[index[show_type][show_ind]].real,y2_semi[index[show_type][show_ind]].real)
    y_total_MPC=np.append(y1_MPC[index[show_type][ind1,ind2]],y2_MPC[index[show_type][ind1,ind2]])
    
    plt.figure()
    #plt.subplot(211)
    plt.plot(t_total, y_total, label="Qutip exact $Re <S^{}_{}S^{}_{}>$".format(show_type[0], ind1,show_type[1],ind2))
   # plt.plot(np.append(t1_semi,t2_semi), y_total_semi, label="1st order $Re <S^{}_{}>$".format(show_type, show_ind))
    plt.plot(np.append(t1_MPC,t2_MPC), y_total_MPC, label="2nd order $Re<S^{}_{}S^{}_{}>$".format(show_type[0], ind1,show_type[1],ind2))
    plt.ylabel("site {}".format(ind1))
    #plt.ylim(-0.6,0.6)
    plt.axhline(y=-0.5, color='grey', linestyle='--')
    plt.legend()
    plt.xlabel('t')
    plt.suptitle('XY model L=%d, N=%d  eps=%.2f  t=%.2f W  g=%.1f W' % (L,N, eps[ind1],t,G))
    if save:
        plt.savefig(path+"/site {}.png".format(double_ind))           
        
    
     
for show_ind in range(N):
#    plot_evolution('+',show_ind)
    plot_evolution(single_type, show_ind)       
    
for show_ind in range(N):
    plot_double_evolution(double_type, show_ind,show_ind-1)