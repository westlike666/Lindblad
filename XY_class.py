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
import copy 
from energy_paras import Energycomputer, Jcomputer, Ucomputer, Gammacomputer


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
    def generate_random_density(self,pure=False,seed=None):   
        
        L=self.L # number of levels 
        N=self.N # number of sites 

        
        state_list=[]

        
        
        for i in (range(N)):
            #np.random.seed(seed)
            state=rand_dm(L,pure=pure,seed=seed)
            #print(state)
            state_list.append(state)     
        rho=tensor(state_list) 
        
        self.random_rho=rho
        return self.random_rho
    
    def generate_coherent_density(self, alpha=np.pi/4, pure=True):   
        
        L=self.L # number of levels 
        N=self.N # number of sites 

        
        state_list=[]
        state=coherent(L, alpha)
        print(state)
        for i in (range(N)):
            state_list.append(state)     
        rho=tensor(state_list) 
        
        self.coherent_rho=rho
        return self.coherent_rho       
    
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
    
    def generate_SpSm(self, flat=True):      
        L=self.L       
        N=self.N
        ops_list=[]
        
        Sp=self.generate_Sp()
        Sm=self.generate_Sm()
        
        for i in (range(N)):
            if not flat:
                ops_list.append([]) 
            for j in range(N):
                op=Sp[i]*Sm[j] # the [i,j] element is Sp_i*Sm_j
                if not flat:
                    ops_list[i].append(op)
                else:
                    ops_list.append(op)
        self.SpSm=ops_list
        return self.SpSm      
    
    def generate_SpSp(self, flat=True):      
        L=self.L       
        N=self.N
        ops_list=[]
        
        Sp=self.generate_Sp()
        
        for i in (range(N)):
            if not flat:
                ops_list.append([]) 
            for j in range(N):
                op=Sp[i]*Sp[j] # the [i,j] element is Sp_i*Sp_j
                if not flat:
                    ops_list[i].append(op)
                else:
                    ops_list.append(op)
        self.SpSp=ops_list
        return self.SpSp      

    def generate_SmSm(self, flat=True):      
        L=self.L       
        N=self.N
        ops_list=[]
        
        Sm=self.generate_Sm()
        
        for i in (range(N)):
            if not flat:
                ops_list.append([]) 
            for j in range(N):
                op=Sm[i]*Sm[j] # the [i,j] element is Sm_i*Sm_j
                if not flat:
                    ops_list[i].append(op)
                else:
                    ops_list.append(op)
        self.SmSm=ops_list
        return self.SmSm   
 
    def generate_SzSz(self, flat=True):      
        L=self.L       
        N=self.N
        ops_list=[]
        
        Sz=self.generate_Sz()
        
        for i in (range(N)):
            if not flat:
                ops_list.append([]) 
            for j in range(N):
                op=Sz[i]*Sz[j] # the [i,j] element is Sz_i*Sz_j
                if not flat:
                    ops_list[i].append(op)
                else:
                    ops_list.append(op)
        self.SzSz=ops_list
        return self.SzSz
    
    def generate_SpSz(self, flat=True):      
        L=self.L       
        N=self.N
        ops_list=[]
        
        Sp=self.generate_Sp()
        Sz=self.generate_Sz()
        
        for i in (range(N)):
            if not flat:
                ops_list.append([]) 
            for j in range(N):
                op=Sp[i]*Sz[j] # the [i,j] element is Sp_i*Sz_j
                if not flat:
                    ops_list[i].append(op)
                else:
                    ops_list.append(op)
        self.SpSz=ops_list
        return self.SpSz

    def generate_SmSz(self, flat=True):      
        L=self.L       
        N=self.N
        ops_list=[]
        
        Sm=self.generate_Sm()
        Sz=self.generate_Sz()
        
        for i in (range(N)):
            if not flat:
                ops_list.append([]) 
            for j in range(N):
                op=Sm[i]*Sz[j] # the [i,j] element is Sm_i*Sz_j
                if not flat:
                    ops_list[i].append(op)
                else:
                    ops_list.append(op)
        self.SmSz=ops_list
        return self.SmSz
    
    def str2op(self, s):
        if s=='z':
            return self.generate_Sz()
        elif s=='+':
            return self.generate_Sp()
        elif s=='-':
            return self.generate_Sm()
        else:
            print('spin operator is not difined')
            
      
    def generate_single_ops(self, single_ops):
        e_ops=[]
        for s in single_ops:
            S=self.str2op(s)
            e_ops += S
        return e_ops    
      
        
    def generate_double_ops(self, double_ops, flat=True):
        N=self.N
        e_ops=[]
        for ss in double_ops:
            S1=self.str2op(ss[0])
            S2=self.str2op(ss[1])
            S1S2=[]
            for i in (range(N)):
                if not flat:
                    S1S2.append([]) 
                for j in range(N):
                    op=S1[i]*S2[j] # the [i,j] element is Sm_i*Sz_j
                    if not flat:
                        S1S2[i].append(op)
                    else:
                        S1S2.append(op)
            e_ops += S1S2
        return e_ops     
        
        
    
    
    

        
    def get_Hamiltonian(self, W=1, t=1, u=0, seed=None):
         L=self.L
         N=self.N
         
         H=Qobj()
         Sz=self.generate_Sz()
         Sp=self.generate_Sp()
         Sm=self.generate_Sm()
         
         #np.random.seed(seed)
         eps=Energycomputer(N,seed).uniformrandom_e(W)
         J=Jcomputer(N, nn_only=False, scaled=False, seed=seed).uniformrandom_j(t)
         U=Ucomputer(N, nn_only=False, scaled=True, seed=seed).uniformrandom_u(u)
         
         self.eps=eps
         self.J=J
         self.U=U
         
         for i in range(N):
             H += eps[i]*Sz[i]    
             for j in range(N):
                 H = H + J[i,j]*(Sp[i]*Sm[j]+Sp[j]*Sm[i])+ U[i,j]*Sz[i]*Sz[j]
         self.Halmitonian=H        
         return H  
     
    def generate_gamma(self, G=1):   
        self.gamma=Gammacomputer(self.N).uniformrandom_g(G)
        return self.gamma
        

    def get_Hamiltonian2(self, eps, J, U):
         """
         """
         L=self.L
         N=self.N
         
         H=Qobj()
         Sz=self.generate_Sz()
         Sp=self.generate_Sp()
         Sm=self.generate_Sm()
         
         #np.random.seed(seed)
         # eps=eps_comp
         # J=J_comp
         # U=U_comp
         
         self.eps=eps
         self.J=J
         self.U=U
         
         for i in range(N):
             H += eps[i]*Sz[i]    
             for j in range(N):
                 H = H + J[i,j]*(Sp[i]*Sm[j]+Sp[j]*Sm[i])+ U[i,j]*Sz[i]*Sz[j]
         self.Halmitonian=H        
         return H  

         
    def generate_gamma2(self, G_comp):   
        self.gamma=G_comp
        return self.gamma          


  
        
            
            
        
        
        
        
          
          
          
          
          
          
 
      
    
          
          
          
          
          
      
        
    
    
    
    
    
    
    
    
    
    
    