# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:38:06 2021

This code is to simulate the Bose-Hubbard model with semi-classical approximation
recource: PLA 91, 033823(2015)

@author: User
"""
import numpy as np
#import matplotlib.pyplot as plt
#import scipy
from qutip import*
from tqdm import tqdm

J=0 #hopping 
w=3+2*J #detunning 
U=-1 # onsite repulsion 
A=2 #external driving 
gamma=2 #losses 


class Bose_Hubbard():
    
    def __init__(self, num_excitation, num_site):
        
        self.N=num_excitation
        self.L=num_site
        
        

    def generate_initial_density(self):
        # N is the dimension of each fock states (number of max excitation)
        # L is the number of sites
        # rho is the total density matrix with dimension (N*L)^2        
        N=self.N
        L=self.L
        
        state_list=[]
        state=thermal_dm(N,1)
        for i in tqdm(range(L)):
            state_list.append(state)     
        rho=tensor(state_list)  
        self.rho=rho
        return self.rho
            
    
    
    def generate_ladder_ops(self):  

        N=self.N
        L=self.L
        
        a=destroy(N)
        adag=create(N)
        
        a_list=[]
        adag_list=[]
            
        
        for i in tqdm(range(L)):
            op_list=[]
            for m in range(L):
                op_list.append(qeye(N))
                
            op_list[i]=a
            a_list.append(tensor(op_list))
            
            op_list[i]=adag
            adag_list.append(tensor(op_list))  
            self.a_list=a_list
            self.adag_list=adag_list
            
        return a_list, adag_list 
    
    
    def generate_adag_a_ops(self, flat=True): # the default is flatten the a_i*a_j row by row
        N=self.N
        L=self.L    
        
        ops_list=[]
        
        a_list, adag_list = self.generate_ladder_ops()
        
        for i in tqdm(range(L)):
            if not flat:
                ops_list.append([]) 
            for j in range(L):
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
        
        a_list, adag_list = self.generate_ladder_ops()
        
        for i in tqdm(range(L)):
            if not flat:
                ops_list.append([]) 
            for j in range(L):
                op=a_list[i]*a_list[j] # the [i,j] element is adag_i*a_j
                if not flat:
                    ops_list[i].append(op)
                else:
                    ops_list.append(op)
        self.a_a_list=ops_list
        return self.a_a_list
    
    
    def get_init_value(self): # get the expectation value of the L+L^2+L^2 opeartors 
        N=self.N
        L=self.L       
        
        rho=self.generate_initial_density()
        
        a_list, _ = self.generate_ladder_ops()
        
        adag_a_list=self.generate_adag_a_ops()
        
        a_a_list=self.generate_a_a_ops()
        
        all_ops=a_list+adag_a_list+a_a_list # append the a, adag*a, a*a operators 
        
        self.init_value=expect(all_ops, rho)
        
        return self.init_value
    
    
    def get_index(self): #get the index of matrix a, adag*a, a*a in the flatten vector 
        L=self.L
        ind_a=[]
        ind_adag_a=[]
        ind_a_a=[]
        
        for i in tqdm(range(L)):
            
            ind_a.append(i)
            ind_adag_a.append([])
            ind_a_a.append([])
            
            for j in range(L):
                ind_adag_a[i].append(L+i*L+j)
                ind_a_a[i].append(L+L*L+i*L+j)
                
        return ind_a, ind_adag_a, ind_a_a  
    
    
    def get_Hamiltonian(self):
        N=self.N
        L=self.L
        
        a, adag=self.generate_ladder_ops()  
        H=Qobj()
        
        for i in tqdm(range(L)):       
            H += w*adag[i]*a[i]+U/2*adag[i]**2*a[i]**2-J*(adag[(i+1)%L]*a[i]+adag[i]*a[(i+1)%L])+A*(adag[i]+a[i])        
        return H  
  
            
            
            
    
    
    
    
                
                
                
        
        
    
    







            
    
    

















