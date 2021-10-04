# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 16:52:46 2021

This code is to simulate the XY model for ultracold plasma
 
John Sous and Edward Grant 2019 New J. Phys. 21 043033

@author: westl
"""

import numpy as np
#import matplotlib.pyplot as plt
#import scipy
from qutip import*
from tqdm import tqdm




class XY():
    def __init__(self,L=2,N=1):
        """
        Parameters
        ----------
        L : int
            number of levels. For spin 1/2 system, L=2
        N : int
            number of sites
        """
        self.L=L
        self.N=N
        self.s=(L-1)/2
    def generate_random_density(self):   
        
        L=self.L # number of levels 
        N=self.N # number of sites 

        
        state_list=[]
        state=rand_dm(L,pure=True)
        
        for i in (range(N)):
            state_list.append(state)     
        rho=tensor(state_list) 
        
        self.random_rho=rho
        return self.random_rho   
    
    def generate_Sz(self): 
        """
        generate a list of S_z operators
        
        """
        L=self.L
        N=self.N
        
        s=(L-1)/2
        
        spin=spin_Jz(s)
        Sz=[]
            
        
        for i in (range(N)):
            op_list=[]
            for m in range(N):
                op_list.append(qeye(L))
                
            op_list[i]=spin
            Sz.append(tensor(op_list))
        self.Sz=Sz
        return Sz
    
    
    def generate_Sp(self):
        """
        generate a list of S_+ operators
        
        """        
        L=self.L
        N=self.N
        
        s=(L-1)/2
        
        spin=spin_Jp(s)
        Sp=[]
            
        
        for i in (range(N)):
            op_list=[]
            for m in range(N):
                op_list.append(qeye(L))
                
            op_list[i]=spin
            Sp.append(tensor(op_list))
        self.Sp=Sp
        return Sp
        
    def generate_Sm(self):
        """
        generate a list of S_- operators
        
        """        
        L=self.L
        N=self.N
        
        s=(L-1)/2
        
        spin=spin_Jm(s)
        Sm=[]
            
        
        for i in (range(N)):
            op_list=[]
            for m in range(N):
                op_list.append(qeye(L))
                
            op_list[i]=spin
            Sm.append(tensor(op_list))
        self.Sm=Sm
        return Sm   

        
    def get_Hamiltonian(self):
         L=self.L
         N=self.N
         
         H=Qobj()
         Sz=self.generate_Sz()
         Sp=self.generate_Sp()
         Sm=self.generate_Sm()
         
         eps=Energycomputer(N).uniformrandom_e(1)
         J=Jcomputer(N).constant_j(1)
         U=Ucomputer(N).constant_u(0)
         
         self.eps=eps
         self.J=J
         self.U=U
         
         for i in range(N):
             H += eps[i]*Sz[i]
             for j in range(N):
                 H = H + J[i,j]*(Sp[i]*Sm[j]+Sp[j]*Sm[i])+ U[i,j]*Sz[i]*Sz[j]
         self.Halmitonian=H        
         return H        
         
         
        
    
class Energycomputer():
    def __init__(self,N):
        """
        generate a list of onsite energies 
        """
        
        self.N=N 
        self.rng=np.random.default_rng() #random number generator 
        
        
    def constant_e(self, E=1):
        energies=E*np.ones(self.N)
        return energies 
    
    
    def uniformrandom_e(self, W=1):  
        """
        ----------
        W : float,
            magnitude of energy disorder. The default is 1.

        Returns
        -------
        energies : array
            onsite energies chosen from a uniform random distribution between -W/2 and W/2
        """
        energies = self.rng.uniform(-W/2, W/2, self.N)
        return energies
    
    
    
 
class Jcomputer():
      def __init__(self, N, nn_only=False, scaled=True):
          """
          ----------
          N : int. numbwer of sites .
          nn_only : Boolean. Whether consider nearest neighbours only. The default is False.
          scaled : Boolean. Whether scale J as J_ij = t/r^3.  The default is True.
          
          Return a matrix of spin-exchanging term J
          """
          self.N=N
          self.nn_only=nn_only 
          self.scaled=scaled
          self.rng=np.random.default_rng()
          
      def jfinder(self,t,x1,x2):
          """
          t is the amplitude J_ij, x1=i, x2=j 
          """
          r=abs(x1-x2)
          if x1==x2:
              return 0
          elif self.nn_only and r>1:
              return 0
          else:
              return t/(r*r*r) if self.scaled else t
          
      def constant_j(self,t):
          N=self.N
          J=t*np.ones((N,N))
          np.fill_diagonal(J,0)
          if self.scaled:
              for i in range(N):
                  for j in range(N):
                      J[i,j]=self.jfinder(J[i,j], i, j)
          return J            
                      
            
      def uniformrandom_j(self, J_max):
          """
          return a random uniform J matrix
          """
          N=self.N
          A=self.rng.uniform(0, J_max, (N,N)) # Can not handle size over N=10^4 !
          J=(A+A.T)/2
          np.fill_diagonal(J,0)
          if self.scaled:
              for i in range(N):
                  for j in range(N):
                      J[i,j]=self.jfinder(J[i,j], i, j)
          if (J==J.T).all():
              return J



class Ucomputer():

      def __init__(self, N, nn_only=True, scaled=True):
          """
          ----------
          N : int. numbwer of sites .
          nn_only : Boolean. Whether consider nearest neighbours only. The default is True.
          scaled : Boolean. Whether scale U as U_ij = u/r^6.  The default is True.
          
          Return a matrix of the Ising term U
          """
          self.N=N
          self.nn_only=nn_only 
          self.scaled=scaled
          self.rng=np.random.default_rng()
          
      def ufinder(self,u,x1,x2):
          """
          u the amplitude of U_ij, x1=i, x2=j 
          """
          r=abs(x1-x2)
          if x1==x2:
              return 0
          elif self.nn_only and r>1:
              return 0
          else:
              return u/(r**6) if self.scaled else u
          
          
      def constant_u(self,u):
          N=self.N
          U=u*np.ones((N,N))
          np.fill_diagonal(U,0)
          if self.scaled:
              for i in range(N):
                  for j in range(N):
                     U[i,j]=self.ufinder(U[i,j], i, j)
          return U            
                      
            
      def uniformrandom_u(self, U_max):
          """
          return a random uniform J matrix
          """
          N=self.N
          A=self.rng.uniform(0, U_max, (N,N)) # Can not handle size over N=10^4 !
          U=(A+A.T)/2
          np.fill_diagonal(U,0)
          if self.scaled:
              for i in range(N):
                  for j in range(N):
                      U[i,j]=self.ufinder(U[i,j], i, j)
          if (U==U.T).all():
              return U            
          
class ode_funs():
    
    def __init__(self, N, eps, J, U, Diss='S_z'):
        """
        Parameters
        ----------
        N : int. number of sites.
        eps: list of onsite energies
        J: matrix of flip-flop term J
        U: matrix of Ising term
        Diss : str, optional
            Choose the type of dissipator. S_z for dephasing and S_m for disspipation. The default is 'S_z'.

        Returns
        -------
        a function fun(t,Y,arg) to feed into the ode solver. 
        """
        self.N=N
        self.eps=eps
        self.J=J
        self.U=U
        self.Diss=Diss
        
    def test(self,t,Y):
        D=self.eps*Y
        return D
    
    def fun_1st(self,t,Y):
        """
        Parameters
        ----------
        t : TYPE
            DESCRIPTION.
        Y : shape of (2,N) for Sz list and Sp list respectively 

        Returns
        -------
        D: shape of (2,N) Derivative of Sz and Sp
        """
        
        D=0*Y+0j
        
    def flat_index(self):
        N=self.N
        
        #use hash table for a input  ['sz','sp'] ['s1s2', 's2s3', ...]
        
        
        
        
          
          
          
          
          
          
 
      
    
          
          
          
          
          
      
        
    
    
    
    
    
    
    
    
    
    
    