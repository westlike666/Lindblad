# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:14:44 2021

@author: User
"""

import numpy as np
import random
from XY_class import*
from XY_ode_funs import ode_funs
from energy_paras import Energycomputer, Jcomputer, Ucomputer, Gammacomputer
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time 
from datetime import datetime 
from tqdm import tqdm
from qutip import*
import argparse

def main():
    L=2
    N=1
    
    W=1
    t=1
    u=0
    G1=0.0
    G2=1
    
    t_0=0
    t_1=100
    steps=1000
    
    t_2=200
    
    
    seed=1
    
    model=XY(L,N)
    
    eps=Energycomputer(N,seed).uniformrandom_e(W)
    J=Jcomputer(N, nn_only=False, scaled=False, seed=seed).uniformrandom_j(t)
    U=Ucomputer(N, nn_only=False, scaled=True, seed=seed).uniformrandom_u(u)
    gamma=Gammacomputer(N).constant_g(G1) 
        
    rng=np.random.default_rng(seed=seed)    
    
    
    #Diss='dephasing' 
    Diss='dissipation'
    
    """
    sloving by semi-classical 1st order: <S1*S2>=<S1>*<S2> for the first stage with decay G1
    """
    

    ode_class=ode_funs(N, eps, J, U, gamma, Diss=Diss) # chose the jump opperator for 'dephasing' or 'dissipation'
    
    fun=ode_class.fun_1st
    
    index=ode_class.flat_index(single_ops=['z', '+'], double_ops=[], index={}) 
    
    def initialize(method):
        
        if method=='y':
            y=np.zeros(2*N)+0j
            y[index['z']]=-0.4 #rng.uniform(-0.5, 0.5)
            y[index['+']]=0.3  #rng.uniform(-0.5, 0.5)
            
        elif method=='rho':
            rho0=model.generate_coherent_density(alpha=1*np.pi/2.5)
            #rho0=model.generate_random_density(seed=None)
            #print(rho0)
            Sz=model.generate_Sz()
            Sp=model.generate_Sp()
            
            e_ops=Sz+Sp   
            y=expect(e_ops, rho0)            
        else:
            print('methode is not difeined')
           
        return y      
    
   
    y1=initialize(method='y')
    
    t_span=(t_0,t_1)
    t_eval=np.linspace(t_0, t_1, steps)
    
    with tqdm(total=100, unit="‰") as pbar:
        result1=solve_ivp(fun, t_span=t_span, y0=y1, t_eval=t_eval, args=[index, pbar, [t_0, (t_1-t_0)/100]])  
    
    
        
    """
    sloving by semi-classical 1st order: <S1*S2>=<S1>*<S2> for the second stage with decay G2
    """
    
    y2=result1.y[:,-1]
    
    
    
    gamma=Gammacomputer(N).boundary_g(G2)
    ode_class=ode_funs(N, eps, J, U, gamma, Diss=Diss)
    fun=ode_class.fun_1st
    
    t_span=(t_1, t_2)
    t_eval=np.linspace(t_1, t_2, steps)
    
    with tqdm(total=100, unit="‰") as pbar:
        result2=solve_ivp(fun, t_span=t_span, y0=y2, t_eval=t_eval, args=[index, pbar, [t_1, (t_2-t_1)/100]]) 
    
    y3=result2.y[:,-1]
    
    """
    plotting the total evolution
    """
    
    def plot_evolution(show_type='z', show_ind=0):

        
        t_total=np.append(result1.t,result2.t)
        y_total=np.append(result1.y[index[show_type][show_ind]],result2.y[index[show_type][show_ind]])
        
        plt.figure()
        plt.subplot(211)
        plt.plot(t_total, y_total, label='Re')  
        plt.ylabel("$<S^{}_{}>$".format(show_type, show_ind))
        plt.legend() 
        plt.subplot(212)
        plt.plot(t_total, y_total.imag, label='Im')    
        plt.xlabel('t')
        plt.ylabel("$<S^{}_{}>^*$".format(show_type, show_ind))
        plt.legend()  
        plt.suptitle('XY model L=%d, N=%d  t=%.1f W  g=%.1f W for ' % (L,N,t,G2)+Diss)
    
    show_ind=random.randrange(N)
    #show_ind=1
    plot_evolution('z',show_ind)
    plot_evolution('+',show_ind)
    
    
    
    
    
    plt.figure()
    plt.plot(y1,'o', label='y1')
    plt.plot(y2, 'x', label='y2')
    plt.plot(y3, '^', label='y3')
    plt.legend()

main()    

            
            
            
        
        
        
        
        