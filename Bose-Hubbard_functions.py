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


def generate_adag_a_ops(N,L,flat=True): # the default is flatten the a_i*a_j row by row
    ops_list=[]
    
    a_list, adag_list = generate_ladder_ops(N, L)
    
    for i in range(L):
        if not flat:
            ops_list.append([]) 
        for j in range(L):
            op=adag_list[i]*a_list[j] # the [i,j] element is adag_i*a_j
            if not flat:
                ops_list[i].append(op)
            else:
                ops_list.append(op)
    return ops_list


def generate_a_a_ops(N,L,flat=True): #the default is flatten the a_i*a_j row by row
    ops_list=[]
    
    a_list, adag_list = generate_ladder_ops(N, L)
    
    for i in range(L):
        if not flat:
            ops_list.append([]) 
        for j in range(L):
            op=a_list[i]*a_list[j] # the [i,j] element is adag_i*a_j
            if not flat:
                ops_list[i].append(op)
            else:
                ops_list.append(op)
    return ops_list


def get_init_value(N,L): # get the expectation value of the L+L^2+L^2 opeartors 
    rho=generate_initial_density(N, L) 
    
    a_list, _ =generate_ladder_ops(N, L)
    
    adag_a_list=generate_adag_a_ops(N, L)
    
    a_a_list=generate_a_a_ops(N, L)

    all_ops=a_list+adag_a_list+a_a_list # append the a, adag*a, a*a operators 
    
    return expect(all_ops, rho)
    







            
    
    

















