# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 15:17:10 2021

@author: User
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from tqdm import tqdm
from qutip import*

def Liouvillian_ode(t, Y, H, c_ops, pbar, state):
    """
    The function feeding into the ode solver
    """
    last_t, dt = state
    n = int((t - last_t)/dt)
    pbar.update(n)
    state[0] = last_t + dt * n
    
    L=liouvillian(H, c_ops, data_only=True)
    
    return L*Y

def Lindblad_solve(H, rho0, t_span, t_eval=None, c_ops=[], e_ops=[]):
    
    vec_rho0=operator_to_vector(rho0)
    d=vec_rho0.dims
    s=vec_rho0.shape
    t=vec_rho0.type
    
       
    y0=np.array(vec_rho0.get_data().todense()).squeeze()
    t_0,t_1=t_span
    fun=Liouvillian_ode    
    with tqdm(total=100) as pbar:
        result=integrate.solve_ivp(fun=fun, t_span=t_span, t_eval=t_eval, y0=y0, args=[H, c_ops, pbar,[t_0, (t_1-t_0)/100]])
    expect_value=[]
    j=0
    while e_ops:
        e_op=e_ops.pop(0)
        expect_value.append([])
        for i , _ in enumerate(t_eval):
            y=Qobj(result.y[:,i].T,dims=d, shape=s, type=t)
            rho_t=vector_to_operator(y)
            
            expect_value[j].append(expect(e_op,rho_t))
        j+=1
                
    return result, expect_value     
 

# H=sigmaz()
# c_ops=sigmax()
# rho0=coherent_dm(2, np.pi/4) 
  
# t0=0
# t1=1
# t_span=(t0, t1)
# t_eval=np.linspace(t0, t1, 100)
 
# result, expect_value=Lindblad_solve(H, rho0, t_span, t_eval, c_ops=[sigmam()], e_ops=[sigmax(),sigmaz()])    
 
   
    
    
    


#Rho=operator_to_vector(rho).get_data()



#L=liouvillian(H,c_ops,data_only=True)

#D=L*Rho

       