# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 16:04:39 2021

@author: westl
"""

import numpy as np
from Bose_Hubbard_class import Bose_Hubbard
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
#from qutip import*

N=2 # max number excitation
L=10**3 # number of site   

"""
for two-level system, the maxmum site the computer can handle for cnstructing initial matrix is about 20 

for 1 billion entries it takes 10GB memory. so do not exceed 10^5 sites 

""" 

J=0 #hopping 
w=3+2*J #detunning 
U=0 # onsite repulsion 
A=2 #external driving 
gamma=2 #losse



model=Bose_Hubbard(N,L)
ind_a, ind_adag_a, ind_a_a = model.get_index()
#y0=model.get_init_value()
y0=np.random.randn(2*L**2+L)+(np.random.randn(2*L**2+L))*1j

def f(t,Y):
    D=0*Y+0j
    
    for l in range(L):
            #assuming the periodic boundary conditions, the index has to take modluo L   
            
            f_a=((w-1j*gamma/2)*Y[ind_a[l]]+A-J*(Y[ind_a[(l+1)%L]]+Y[ind_a[(l-1)%L]]))/1j
            # +U*(2*Y[ind_adag_a[l][l]]*Y[ind_a[l]]+\
            #     Y[ind_a_a[l][l]]*np.conjugate(Y[ind_a[l]])\
            #     -2*(Y[ind_a[l]])**2*np.conjugate(Y[ind_a[l]]))
            
            D[ind_a[l]]=f_a
            
            for m in range(L):
                
                f_b=(-1j*gamma*Y[ind_adag_a[l][m]]+A*(np.conjugate(Y[ind_a[l]])-Y[ind_a[m]])\
                    +J*(Y[ind_adag_a[(l-1)%L][m]]+Y[ind_adag_a[(l+1)%L][m]]\
                        -Y[ind_adag_a[l][(m-1)%L]]-Y[ind_adag_a[l][(m+1)%L]]))/1j        
                
                f_c=((2*w-1j*gamma)*Y[ind_a_a[l][m]]+A*(Y[ind_a[l]]+Y[ind_a[m]])\
                    -J*(Y[ind_a_a[(l-1)%L][m]]+Y[ind_a_a[(l+1)%L][m]]\
                        +Y[ind_a_a[l][(m-1)%L]]+Y[ind_a_a[l][(m+1)%L]]))/1j
                
                D[ind_adag_a[l][m]]=f_b
                D[ind_a_a[l][m]]=f_c
    return D   



t_span=(0,10)
#t_eval=np.linspace(0, 10, 100)
result=solve_ivp(f, y0=y0, t_span=t_span) 
plt.plot(result.t, result.y[0])     
                    
            
                
            
                
                
                
                
                    
                    
                    
                    