# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 14:38:06 2021

This code is to simulate the Bose-Hubbard model with semi-classical approximation
recource: PLA 91, 033823(2015)

@author: User
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy
from qutip import*

J=0 #hopping 
w=3+2*J #detunning 
U=-1 # onsite repulsion 
A=2 #external driving 
gamma=2 #losses 



def generate_initial_density(N,L):
    
    state_list=[]
    state=thermal_dm(N,1)
    for i in range(L):
        state_list.append(state)     
    rho=tensor(state_list)  
    return rho
        




def generate_ladder_ops(N,L):  
    # N is the dimension of each fock states (number of max excitation)
    # L is the number of sites
    # rho is the total density matrix with dimension (N*L)^2
    a=destroy(N)
    adag=create(N)
    
    a_list=[]
    adag_list=[]
        
    
    for i in range(L):
        op_list=[]
        for m in range(L):
            op_list.append(qeye(N))
            
        op_list[i]=a
        a_list.append(tensor(op_list))
        
        op_list[i]=adag
        adag_list.append(tensor(op_list))   
        
    return a_list, adag_list 


def generate_adag_a_ops(N,L):
    matrix=[]
    
    a_list, adag_list = generate_ladder_ops(N, L)
    
    for i in range(L):
        matrix.append([]) 
        for j in range(L):
            op=adag_list[i]*a_list[j] # the [i,j] element is adag_i*a_j
            matrix[i].append(op)           
    return matrix


def generate_a_a_ops(N,L):
    matrix=[]
    
    a_list, adag_list = generate_ladder_ops(N, L)
    
    for i in range(L):
        matrix.append([]) 
        for j in range(L):
            op=a_list[i]*a_list[j] # the [i,j] element is a_i*a_j
            matrix[i].append(op)          
    return matrix





            
    
    

















