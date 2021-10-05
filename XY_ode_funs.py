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
        
    def test(self,t,Y):
        D=self.eps*Y
        return D
    
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
        
        D=0*Y+0j
        
        for l in range(self.N):
            f1=0
            f2=1j*self.eps[l]*Y[index['+'][l]]
            for i in range(self.N):
               f1 += 2j*self.J[i,l]*(Y[index['+'][i]]*Y[index['+'][l]].conjugate()\
                   -Y[index['+'][l]]*Y[index['+'][i]].conjugate())
               f2 += -4j*self.J[i,l]*Y[index['+'][i]]*Y[index['z'][l]]\
                   -2j*self.U[i,l]*Y[index['+'][l]]*Y[index['z'][i]]    
                   
            if self.Diss=='dephasing':
                g1=0
                
                g2=-1/2*self.gamma[l]*Y[index['+'][l]]
                
            elif self.Diss=='dissipation':
                g1=-self.gamma[l]*Y[index['+'][l]]*Y[index['+'][l]].conjugate()
                g2=self.gamma[l]*Y[index['+'][l]]*Y[index['z'][l]]
                
            else:
                print('Jump operator is not difined')
            
            D[index['z'][l]] = f1+g1
            D[index['+'][l]] = f2+g2
            
        return D    
            
              
    
    def flat_index(self, single_ops, double_ops, index):
        """        
        Parameters
        ----------
        single_ops : list of str e.g. ['z', '+']
        
        double_ops : list of str. e.g.  ['z+', '+-', ...]

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
            print(pre)
            value=np.array([range(N*i+pre, N*(i+1)+pre)  for i in range(N)])
            pre += N**2
            index[key]=value
                  
        return index   