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
#from tqdm import tqdm
import copy 
from energy_paras import Energycomputer, Jcomputer, Ucomputer, Gammacomputer


class HCB():
    def __init__(self,L=2,N=1):
        """
        Parameters
        ----------
        L : int
            number of levels. For Hard-core Bosons, L=2. 
        N : int
            number of sites
        """
        self.L=L
        self.N=N
#        self.s=(L-1)/2
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
        return state_list,self.random_rho
    
    def generate_random_ket(self,seed=None):   
        
        L=self.L # number of levels 
        N=self.N # number of sites 
        
        state_list=[]

        for i in (range(N)):
            #np.random.seed(seed)
            state=rand_ket(L,seed=seed)
            #print(state)
            state_list.append(state)     
        rho=tensor(state_list) 
        
        self.random_rho=rho
        return state_list,self.random_rho    
    
    def generate_one(self, up_sites):
        L=self.L
        N=self.N
        
        state_list=[]
        
        for i in range(N):
            state=fock_dm(L,0)
            state_list.append(state)
        for i in up_sites:    
            state_list[i]=fock_dm(L,1)
        rho=tensor(state_list)        
        return state_list, rho
    
    def generate_coherent_density(self, alpha=np.pi/4, pure=True):   
        
        L=self.L # number of levels 
        N=self.N # number of sites 

        
        state_list=[]
        state=coherent(L, alpha)
        #print(state)
        for i in (range(N)):
            state_list.append(state)     
        rho=tensor(state_list) 
        
        self.coherent_rho=rho
        return state_list, self.coherent_rho       
    
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
    
    
    def generate_a(self):
        """
        generate a list of anhilation operators
        
        """        
        L=self.L
        N=self.N
                
        ladder=destroy(L)
        a=[]
            
        
        for i in (range(N)):
            op_list=[]
            for m in range(N):
                op_list.append(qeye(L))
                
            op_list[i]=ladder
            a.append(tensor(op_list))
        self.a=a
        return a
        
    def generate_adag(self):
        """
        generate a list of creation operators
        
        """        
        L=self.L
        N=self.N
                
        ladder=create(L)
        adag=[]
            
        
        for i in (range(N)):
            op_list=[]
            for m in range(N):
                op_list.append(qeye(L))
                
            op_list[i]=ladder
            adag.append(tensor(op_list))
        self.adag=adag
        return adag

    def generate_num(self):
        """
        generate a list of anhilation operators
        
        """        
        L=self.L
        N=self.N
                
        ladder=num(L)
        number=[]
            
        
        for i in (range(N)):
            op_list=[]
            for m in range(N):
                op_list.append(qeye(L))
                
            op_list[i]=ladder
            number.append(tensor(op_list))
        self.number=number    
        return number



    def generate_adag_a_ops(self, flat=True): # the default is flatten the a_i*a_j row by row
        N=self.N
        L=self.L    
        
        ops_list=[]
        
        a_list=self.generate_a()
        adag_list=self.generate_adag()    
        
        for i in (range(N)):
            if not flat:
                ops_list.append([]) 
            for j in range(N):
                op=adag_list[i]*a_list[j] # the [i,j] element is adag_i*a_j
                if not flat:
                    ops_list[i].append(op)
                else:
                    ops_list.append(op)
        self.adag_a_list=ops_list            
        return self.adag_a_list
    
    
    def generate_a_a_ops(self, flat=True): #the default is flatten the a_i*a_j row by row
        N=self.N
        L=self.L       
    
        ops_list=[]
        
        a_list=self.generate_a()
        adag_list=self.generate_adag()  
        
        for i in (range(N)):
            if not flat:
                ops_list.append([]) 
            for j in range(N):
                op=a_list[i]*a_list[j] # the [i,j] element is adag_i*a_j
                if not flat:
                    ops_list[i].append(op)
                else:
                    ops_list.append(op)
        self.a_a_list=ops_list
        return self.a_a_list
 
 


   
 
    
    
    def str2op(self, s):
        if s=='n':
            return self.generate_num()
        elif s=='+':
            return self.generate_adag()
        elif s=='-':
            return self.generate_a()
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
         num=self.generate_num()
         a=self.generate_a()
         adag=self.generate_adag()
         
         #np.random.seed(seed)
         eps=Energycomputer(N,seed).uniformrandom_e(W)
         J=Jcomputer(N, nn_only=False, scaled=False, seed=seed).uniformrandom_j(t)
         U=Ucomputer(N, nn_only=False, scaled=True, seed=seed).uniformrandom_u(u)
         
         self.eps=eps
         self.J=J
         self.U=U
         
         for i in range(N):
             H += eps[i]*num[i]    
             for j in range(N):
                 H = H + J[i,j]*(adag[i]*a[j]+adag[j]*a[i])+ U[i,j]*num[i]*num[j]
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
         num=self.generate_num()
         a=self.generate_a()
         adag=self.generate_adag()
         
         #np.random.seed(seed)
         # eps=eps_comp
         # J=J_comp
         # U=U_comp
         
         self.eps=eps
         self.J=J
         self.U=U
         
         for i in range(N):
             H += eps[i]*num[i]    
             for j in range(N):
                 H = H + J[i,j]*(adag[i]*a[j]+adag[j]*a[i])+ U[i,j]*num[i]*num[j]
         self.Halmitonian=H       
         return H  

         
    def generate_gamma2(self, G_comp):   
        self.gamma=G_comp
        return self.gamma          


  
        
            
            
        
        
        
        
          
          
          
          
          
          
 
      
    
          
          
          
          
          
      
        
    
    
    
    
    
    
    
    
    
    
    