# -*- coding: utf-8 -*-
"""
Created on Fri Oct 15 15:07:56 2021

@author: User
"""


import numpy as np
#from qutip import*


class Energycomputer():
    def __init__(self, N, seed=None):
        """
        generate a list of onsite energies 
        """
        
        self.N=N 
        self.rng=np.random.default_rng(seed=seed) #random number generator 
        
        
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
        #np.random.seed(seed)
        energies = self.rng.uniform(-W/2, W/2, self.N)
        return energies
    

class Gammacomputer():
    def __init__(self, N, seed=None):
        """
        generate a list of dissipation rate gamma 
        """
        self.N=N 
        self.rng=np.random.default_rng(seed) #random number generator
        
    def constant_g(self, g=1):
        gamma=g*np.ones(self.N)
        return gamma  
    
    def uniformrandom_g(self, G=1, seed=None):

        gamma=self.rng.uniform(0, G, self.N)
        return gamma
    
    def central_g(self, G=1):
        gamma=np.zeros(self.N)
        center=self.N//2
        gamma[center]=G
        return gamma
    
    def boundary_g(self, G=1):
        gamma=np.zeros(self.N)
        gamma[0]=G
        return gamma
        
    
    
    
    
 
class Jcomputer():
      def __init__(self, N, nn_only=False, scaled=False, seed=None):
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
          self.rng=np.random.default_rng(seed=seed)
          
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
    
    

      def __init__(self, N, nn_only=False, scaled=True, seed=None):
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
          self.rng=np.random.default_rng(seed)
          
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