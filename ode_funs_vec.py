# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:49:27 2022

@author: User
"""

import numpy as np
from qutip import*


class ode_funs():
    
    def __init__(self, N, eps, J, U, gamma, Diss='depha'):
        """
        Parameters
        ----------
        N : int. number of sites.
        eps: list of onsite energies
        J: matrix of flip-flop term J
        U: matrix of Ising term
        index: Dictionary 
            Choose the type of dissipator. S_z for dephasing and S_m for disspipation. The default is 'deph'.

        Returns
        -------
        a function fun(t,Y,arg) to feed into the ode solver. 
        """
        self.N=N
        self.eps=eps
        self.J=J
        self.U=U
        self.gamma=gamma
        self.Diss=Diss
        
    
    def fun_1st(self,t,Y, index):
        """
        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        Y : shape of (N,) for Sz list and Sp  

        Returns
        -------
        D: shape of (N,) Derivative of Sz and Sp
        """
        
        # last_t, dt = state
        # n = int((t - last_t)/dt)
        # pbar.update(n)
        # state[0] = last_t + dt * n
    
        
        D=0*Y+0j
        
        for k in range(self.N):
            D[index['z'][k]]=0
            D[index['+'][k]]=1j*self.eps[k]*Y[index['+'][k]]
            D[index['-'][k]]=-1j*self.eps[k]*Y[index['-'][k]] 
            for i in range(self.N):
               if i==k: continue 
               D[index['z'][k]] += 1j*self.J[i,k]*(Y[index['+'][i]]*Y[index['-'][k]]-Y[index['+'][k]]*Y[index['-'][i]])
               D[index['+'][k]] += -2j*self.J[i,k]*Y[index['+'][i]]*Y[index['z'][k]]   
               D[index['-'][k]] += +2j*self.J[i,k]*Y[index['-'][i]]*Y[index['z'][k]]

               
            #dissipation term         
            D[index['z'][k]] += -self.gamma[k]*(Y[index['z'][k]]+0.5)
            D[index['+'][k]] += -self.gamma[k]*Y[index['+'][k]]/2
            D[index['-'][k]] += -self.gamma[k]*Y[index['-'][k]]/2    
        return D 
    
    
def sum2(J,Y,index,pmz):
    vec1=np.expand_dims(Y[index[pmz[0]]],1)
    vec2=np.expand_dims(Y[index[pmz[1]]],1)
    
    s=vec1*J@vec2  # element wise and matrix product
    
    return np.squeeze(s) # reduce t o (N,) dimension
    
def sum3(J,Y, index, pmz):
    vec1=np.expand_dims(Y[index[pmz[0]]],1)
    vec2=np.expand_dims(Y[index[pmz[1]]],1)
    vec3=np.expand_dims(Y[index[pmz[2]]],1) #single vectors 
    

    
    mat_12=Y[index[(pmz[0]+pmz[1])]]
    mat_13=Y[index[(pmz[0]+pmz[2])]]
    mat_23=Y[index[(pmz[1]+pmz[2])]] #double operator matrix
    
    # (k,l,i) assume the third index is always the free index i, sum over i 
    # also assume k goes into the the coefficient J_ki 
    
    
    
    return vec3, mat_23
    
    
    
    
    
    
    
    
    
    