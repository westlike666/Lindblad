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
        self.eps=np.expand_dims(eps,1)
        self.J=J
        self.U=U
        self.gamma=np.expand_dims(gamma,1)
        self.Diss=Diss
        
    

    def fun_1st_vec(self, t,Y, index, pbar, state):
        
        
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
        
        last_t, dt = state
        n = int((t - last_t)/dt)
        pbar.update(n)
        state[0] = last_t + dt * n
    
        
        D=0*Y+0j
        

        D[index['z']]=1j*(sum2(self.J,Y,index,'-+')-sum2(self.J,Y,index,'+-'))-self.gamma*(Y[index['z']]+0.5)
        D[index['+']]=1j*self.eps*Y[index['+']]-2j*sum2(self.J,Y,index,'z+')-0.5*self.gamma*(Y[index['+']])
#        D[index['-']]=-1j*self.eps*Y[index['-']]+2j*sum2(self.J,Y,index,'z-')-0.5*self.gamma*(Y[index['-']])
        D[index['-']]=np.conjugate(D[index['+']])
             
        return D     
    

    def generate_full_op_list(self):
        single_ops=['z','+','-']
        double_ops=[]
        for s1 in single_ops:
            for s2 in single_ops:
                    double_ops.append(s1+s2)
                    
        return single_ops, double_ops 
            
    
    
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
            value_vec=np.reshape(value,(N,1))
            pre += N
            index[key]=value_vec # return colum vector (N,1)
            
        while double_ops:
            key=double_ops.pop(0)
            #print(pre)
            value=np.array([range(N*i+pre, N*(i+1)+pre)  for i in range(N)])
            pre += N**2
            index[key]=value
            self.index=index
                  
        return index       
    
    
    
def sum2(J,Y,index,pmz):
    
    # (k,i) the second is free index !
    
    vec1=Y[index[pmz[0]]]
    vec2=Y[index[pmz[1]]]
    
    s=vec1*J@vec2  # element wise and matrix product
    
    return s # keep (N,1) dimension
    
def sum_ki(J,Y, index, pmz):
    vec1=Y[index[pmz[0]]]
    vec2=Y[index[pmz[1]]]
    vec3=Y[index[pmz[2]]] #single (N,1) vectors 
    

    
    mat12=Y[index[(pmz[0]+pmz[1])]]
    mat13=Y[index[(pmz[0]+pmz[2])]]
    mat23=Y[index[(pmz[1]+pmz[2])]] #double operator matrix
    
    # (k,l,i) assume the third index is always the free index i, sum over i 
    # also assume the first index k goes into the the coefficient J_ki 
     
    s_kl_i=mat12*(J@vec3)
    s_ki_l=np.sum(mat13*J, axis=1,keepdims=True)*vec2.T
    s_li_k=J@mat23.T*vec1
    s_kli=(J@vec3)*(vec1*vec2.T)
    
    
    return s_kl_i+s_ki_l+s_li_k+2*s_kli
    
def sum_li(J,Y, index, pmz):

    r=sum_ki(J, Y, index, pmz)
    
    return r.T 
    
    
    
    
    
    
    
    