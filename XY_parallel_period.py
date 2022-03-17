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
#from tqdm import tqdm
from qutip import*
import random
from numpy.random import default_rng
#from Lindblad_solver import Lindblad_solve
from energy_paras import Energycomputer, Jcomputer, Ucomputer, Gammacomputer
import argparse
import sys

parser = argparse.ArgumentParser(description='Run the model as set up')


parser.add_argument('--N', type=int, nargs='?',
                      help='Number of site', action='store')

parser.add_argument('--W', type=float, nargs='?',
                      help='The degree of disorder', action='store')
parser.add_argument('--t', type=float, nargs='?',
                      help='The J bar', action='store')

parser.add_argument('--u', type=float, nargs='?',
                      help='The u term', action='store')

parser.add_argument('--G', type=float, nargs='?',
                      help='dissipation rate', action='store')

parser.add_argument('--bc', type=str, nargs='?',
                        help = 'type of start decaying: center/boundary',
                        action='store', default="center")

parser.add_argument('--seed', type=int, nargs='?',
                      help='seed for random generator', action='store', default=None)

parser.add_argument('--save', type=bool, nargs='?',
                        help = 'Whether to save results to file',
                        action='store', default=False)



args=parser.parse_args()


save=args.save

L=2

N=args.N
W=args.W
t=args.t
u=args.u
G=args.G
bc=args.bc
seed=args.seed

name='Period N='+str(N)+' W='+ str(W)+' t='+str(t) + ' u='+ str(u) + ' g='+ str(G) +' seed=' +str(seed)

if save:
#    path='results/'+utils.get_run_time()
    path='results/'+bc+'/' +name

    os.mkdir(path)
    
show_type='z'
#show_ind=random.randrange(N)
show_ind=0

model=XY(L,N)


eps=Energycomputer(N,seed).uniformrandom_e(W)
J=Jcomputer(N, nn_only=False, scaled=True, seed=seed).constant_j(t)
U=Ucomputer(N, nn_only=False, scaled=True, seed=seed).constant_u(u)
if bc=='center':
    gamma=Gammacomputer(N).central_g(G)
elif bc=='boundary':
    gamma=Gammacomputer(N).boundary_g(G)

#gamma=Gammacomputer(N).site_g(G,[0,6])
#gamma=Gammacomputer(N).constant_g(G)

H=model.get_Hamiltonian2(eps, J, U)

#states,rho0=model.generate_coherent_density(alpha=1*np.pi/4)
#states,rho0=model.generate_random_density(seed=None, pure=True) #seed works for mixed state
#states,rho0=model.generate_random_ket()
rng=default_rng(seed=seed)
#up_sites=rng.choice(N, N//2, replace=False)
up_sites=[i for i in range(0,N,2)]

states, rho0=model.generate_up(up_sites)

#print(states)
print(up_sites)
print(eps)

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

# for i in range(N):
#     c_ops.append(np.sqrt(gamma[i])*diss[i])
    
t_2=500
t_span=(t_1, t_2)
times2=np.linspace(t_1, t_2, 1000)


args={'T':100} 

def coeff_m(t, args):
    T=args['T']
    gate=np.sin(2*np.pi/T*(t-t_1))>=0
    return 1*gate 

def coeff_p(t, args):
    return 1-coeff_m(t, args)





for i in range(N):
 #   c_ops.append(np.sqrt(gamma[i])*diss[i]) # constant c_ops

    c_ops.append([[np.sqrt(gamma[i])*Sm[i],coeff_m],[np.sqrt(gamma[i])*Sp[i],coeff_p]]) # periodic c_ops
 
    

result2=mesolve(H, rho1, times2, c_ops, args=args,progress_bar=True)
y2=expect(e_ops, result2.states) 


# result3, expect_value=Lindblad_solve(H, rho0, t_span, t_eval, c_ops=c_ops, e_ops=e_ops)  

# plt.plot(result3.t, expect_value[index[show_type][show_ind]], label='solve_ivp solved Lindblad') 



def plot_evolution(show_type='z', show_ind=0):

    t_total=np.append(result1.times, result2.times)
    y_total=np.append(y1[index[show_type][show_ind]].real,y2[index[show_type][show_ind]].real)
    plt.figure(show_ind)
    #plt.subplot(211)
    plt.plot(t_total*t, y_total, label="$Re <S^{}_{}>$".format(show_type, show_ind))
    plt.ylabel("site {}".format(show_ind))
    plt.ylim(-0.5,0.5)
    plt.axhline(y=-0.5, color='grey', linestyle='--')
    plt.legend()
    # plt.subplot(212)
    # plt.plot(t_total, result1.y[index[show_type][show_ind]].imag, label='1st-order approx') 
    # plt.plot(t_total, result2.expect[index[show_type][show_ind]].imag, label='Qutip solved Lindblad')
    # plt.ylabel("$Im <S^{}_{}>$".format(show_type, show_ind))
    # plt.legend()
    plt.xlabel('t * $\overline{J}$')
    plt.suptitle('XY model, N=%d, W=%d,  eps=%.2f,  $\overline{J}$=%.2f  u=%d  $\gamma$=%.1f' % (N,W, eps[show_ind],t,u,gamma[show_ind]))
    if save:
        plt.savefig(path+"/site {}.png".format(show_ind))    
for show_ind in range(N):
    plot_evolution('+',show_ind)
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
                 't1':result1.times,
                 't2':result2.times ,                 
                 'index': index,
                 'up_sites':up_sites
                 }
    utils.store_vars(path, var_to_save)



L=liouvillian(H, c_ops, data_only=True)
scipy.sparse.linalg.eigs(L)    
    
    