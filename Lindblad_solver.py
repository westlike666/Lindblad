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

def Liouvillian_ode(t, Y, H, c_ops):
    """
    The function feeding into the ode solver
    """
    
    L=liouvillian(H, c_ops, data_only=True)
    
    return L*Y

def Lindblad_solve(H, rho0, t_span, t_eval=None, c_ops=[], e_ops=[]):
    
    b=operator_to_vector(rho0).get_data().todense()
    y0=np.array(b).squeeze()
    fun=Liouvillian_ode    
    result=integrate.solve_ivp(fun=fun, t_span=t_span, t_eval=t_eval, y0=y0, args=[H, c_ops])
    
    return result     
 

H=sigmaz()
c_ops=sigmax()
rho0=coherent_dm(2, np.pi/4) 
  
t0=0
t1=1
t_span=(t0, t1)
t_eval=np.linspace(t0, t1, 100)
 
result=Lindblad_solve(H, rho0, t_span, t_eval)    
 
   
    
    
    


#Rho=operator_to_vector(rho).get_data()



#L=liouvillian(H,c_ops,data_only=True)

#D=L*Rho

       