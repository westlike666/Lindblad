# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 20:52:18 2021

@author: westl
"""

import numpy as np
#import matplotlib.pyplot as plt
#import scipy
from qutip import*




class ode_funs():
    
    def __init__(self, N, eps, J, U, gamma, Diss='dissipation'):
        """
        Parameters
        ----------
        N : int. number of sites.
        eps: list of onsite energies
        J: matrix of flip-flop term J
        U: matrix of Ising term
        index: Dictionary 
        anhniliation operator for disspipation.

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
        
    def test(self,t,Y):
        D=self.eps*Y
        return D
    
    def fun_1st(self,t,Y, index, pbar, state):
        """
        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        Y : shape of (N,) for a and adag*a  

        Returns
        -------
        D: shape of (N,) Derivative of a and adag*a  
        """
        
        last_t, dt = state
        n = int((t - last_t)/dt)
        pbar.update(n)
        state[0] = last_t + dt * n
    
        
        D=0*Y+0j
        
        for l in range(self.N):
            D[index['-'][l]]=(-1j*self.eps[l]-0.5*self.gamma[l])*Y[index['-'][l]]     # for a        
            
            for i in range(self.N):
                D[index['-'][l]] += -2j*self.J[i,l]*Y[index['-'][i]]
            
                
            for m in range(self.N):
                D[index['+-'][l,m]]=(1j*(self.eps[l]-self.eps[m])-0.5*(self.gamma[l]+self.gamma[m])) *Y[index['+-'][l,m]]  # for adag*a
                
                for i in range(self.N):
                    D[index['+-'][l,m]] += 2j*(self.J[i,l]*Y[index['+-'][i,m]]-self.J[m,i]*Y[index['+-'][l,i]])
                    
                    
        return D    
    
 
            
             
            
            
    def generate_full_op_list(self):
        single_ops=['-','+']
        double_ops=[]
        for s1 in single_ops:
            for s2 in single_ops:
                    double_ops.append(s1+s2)
                    
        return single_ops, double_ops 
            
    
    
    def flat_index(self, single_ops, double_ops, index):
        """        
        Parameters
        ----------
        single_ops : list of str e.g. ['+', '-']
        
        double_ops : list of str. e.g.  ['+-', '--']

        index : dictionary 
            DESCRIPTION. Initiate it to empty {}.

        Returns
        -------
        index : a dictionary contain the index in the flatten vector Y 

        """
        N=self.N       

        pre=0
        while single_ops:
            key=single_ops.pop(0)
            value=np.arange(pre, pre+N)
            pre += N
            index[key]=value
            
        while double_ops:
            key=double_ops.pop(0)
            #print(pre)
            value=np.array([range(N*i+pre, N*(i+1)+pre)  for i in range(N)])
            pre += N**2
            index[key]=value
            self.index=index
                  
        return index   
    
    

    
    
    