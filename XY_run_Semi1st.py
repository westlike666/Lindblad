# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 17:53:05 2021

@author: westl
"""

import scipy
import numpy as np
import os
import utils
from XY_class import*
from XY_ode_funs import ode_funs
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
N=7
W=10
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
#gamma=Gammacomputer(N).site_g(G,[0,6])
#gamma=Gammacomputer(N).constant_g(G)

H=model.get_Hamiltonian2(eps, J, U)

states,rho0=model.generate_coherent_density(alpha=1*np.pi/0.9)
#states,rho0=model.generate_random_density(seed=None, pure=True) #seed works for mixed state
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





"""

"""

   
#Diss='dephasing' 
Diss='dissipation'

# ode_funs=ode_funs(N, eps, J, U, gamma, Diss=Diss) # chose the jump opperator for 'dephasing' or 'dissipation'
# index=ode_funs.flat_index(single_ops=['z','+'], double_ops=[], index={}) 

t_0=0
t_1=100
t_span=(t_0,t_1)
times1=np.linspace(t_0, t_1, 1000)



"""
sloving by Qutip Lindblad

"""

e_ops=Sz+Sp # list of expectation values to evaluate

result1=mesolve(H, rho0, times1, progress_bar=True) 
y1=expect(e_ops, result1.states)


"""
sloving by semi-classical 1st order: <S1*S2>=<S1>*<S2> for the first stage with decay G1
"""



y0=expect(e_ops, rho0)
   


ode_class=ode_funs(N, eps, J, U, gamma=0*gamma, Diss=Diss) # chose the jump opperator for 'dephasing' or 'dissipation'

fun=ode_class.fun_1st

index=ode_class.flat_index(single_ops=['z', '+'], double_ops=[], index={}) 


with tqdm(total=100, unit="‰") as pbar:
    result1_semi=solve_ivp(fun, t_span=t_span, y0=y0, t_eval=times1, args=[index, pbar, [t_0, (t_1-t_0)/100]])  

y1_semi=result1_semi['y']

################################################################################
##############################################################################

"""
Qutip
"""


rho1=result1.states[-1]

if Diss=='dephasing':
    diss=Sz
else:
    diss=Sm
    
c_ops=[]

# for i in range(N):
#     c_ops.append(np.sqrt(gamma[i])*diss[i])
    
t_2=500
t_span=(t_1, t_2)
times2=np.linspace(t_1, t_2, 1000)

for i in range(N):
    c_ops.append(np.sqrt(gamma[i])*diss[i]) # constant c_ops

#    c_ops.append([np.sqrt(gamma[i])*diss[i],np.sin(times2-t_1)/10]) # periodic c_ops
 
    

result2=mesolve(H, rho1, times2, c_ops, args={},progress_bar=True)
y2=expect(e_ops, result2.states) 

"""
semi-classic 1st order
"""
 
ode_class=ode_funs(N, eps, J, U, gamma=gamma, Diss=Diss) # chose the jump opperator for 'dephasing' or 'dissipation'

fun=ode_class.fun_1st
  

with tqdm(total=100, unit="‰") as pbar:
    result2_semi=solve_ivp(fun, t_span=t_span, y0=y1_semi[:,-1], t_eval=times2, args=[index, pbar, [t_1, (t_2-t_1)/100]])  

y2_semi=result2_semi['y']







# result3, expect_value=Lindblad_solve(H, rho0, t_span, t_eval, c_ops=c_ops, e_ops=e_ops)  

# plt.plot(result3.t, expect_value[index[show_type][show_ind]], label='solve_ivp solved Lindblad') 



def plot_evolution(show_type='z', show_ind=0):

    t_total=np.append(result1.times, result2.times)
    y_total=np.append(y1[index[show_type][show_ind]].real,y2[index[show_type][show_ind]].real)   
    y_total_semi=np.append(y1_semi[index[show_type][show_ind]].real,y2_semi[index[show_type][show_ind]].real)
    plt.figure(show_ind)
    #plt.subplot(211)
    plt.plot(t_total, y_total, label="$Qutip Re <S^{}_{}>$".format(show_type, show_ind))
    plt.plot(t_total, y_total_semi, label="$semi-class Re <S^{}_{}>$".format(show_type, show_ind))
    plt.ylabel("site {}".format(show_ind))
    plt.ylim(-0.6,0.6)
    plt.axhline(y=-0.5, color='grey', linestyle='--')
    plt.legend()
    # plt.subplot(212)
    # plt.plot(t_total, result1.y[index[show_type][show_ind]].imag, label='1st-order approx') 
    # plt.plot(t_total, result2.expect[index[show_type][show_ind]].imag, label='Qutip solved Lindblad')
    # plt.ylabel("$Im <S^{}_{}>$".format(show_type, show_ind))
    # plt.legend()
    plt.xlabel('t')
    plt.suptitle('XY model L=%d, N=%d  eps=%.2f  t=%.2f W  g=%.1f W' % (L,N, eps[show_ind],t,G))
    if save:
        plt.savefig(path+"/site {}.png".format(show_ind))    
for show_ind in range(N):
#    plot_evolution('+',show_ind)
    plot_evolution('z', show_ind)
     
    
if save:
    var_to_save={'N':N,
                 'J':J,
                 'U':U,
                 'eps':eps,
                 'gamma':gamma,
                 'states': states,
                 'y1':y1,
                 'y2':y2,
                 'index': index,
                 'up_sites':up_sites
                 }
    utils.store_vars(path, var_to_save)



L=liouvillian(H, c_ops, data_only=True)
scipy.sparse.linalg.eigs(L)    
    
    