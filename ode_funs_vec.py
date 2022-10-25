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
        self.eps=np.reshape(eps,(N,1)) #column (N,1)
        self.J=J
        self.U=U
        self.gamma=np.reshape(gamma,(N,1)) #column (N,1)
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
        

        D[index['z']]=1j*(sum1(self.J,Y,index,'-+')-sum1(self.J,Y,index,'+-'))-self.gamma*(Y[index['z']]+0.5)
        D[index['+']]=1j*self.eps*Y[index['+']]-2j*sum1(self.J,Y,index,'z+')-0.5*self.gamma*(Y[index['+']])
#        D[index['-']]=-1j*self.eps*Y[index['-']]+2j*sum1(self.J,Y,index,'z-')-0.5*self.gamma*(Y[index['-']])
        D[index['-']]=np.conjugate(D[index['+']])
             
        return D  
    
    
    def fun_2nd_vec(self, t, Y, index):  
        
    
        # for key in index:
        #     if len(key)>1:
        #         Y[np.diag(index[key])]=0              
        
        D=0*Y
        
        D[index['z']]=1j*(sum2(self.J,Y,index,'-+')-sum2(self.J,Y,index,'+-'))-self.gamma*(Y[index['z']]+0.5)
        D[index['+']]=1j*self.eps*Y[index['+']]-2j*sum2(self.J,Y,index,'z+')-0.5*self.gamma*(Y[index['+']])
        D[index['-']]=-1j*self.eps*Y[index['-']]+2j*sum2(self.J,Y,index,'z-')-0.5*self.gamma*(Y[index['-']])        
#        D[index['-']]=np.conjugate(D[index['+']])
        
        
        D[index['+-']]=1j*(self.eps-self.eps.T)*Y[index['+-']]-1j*self.J*(Y[index['z']]-Y[index['z']].T)\
                       -2j*(sum_ki(self.J, Y, index, 'z-+')-sum_li(self.J, Y, index, 'z+-'))\
                       -0.5*(self.gamma+self.gamma.T)*Y[index['+-']]    

        D[index['-+']]= -1j*(self.eps-self.eps.T)*Y[index['-+']]+1j*self.J*(Y[index['z']]-Y[index['z']].T)\
                       +2j*(sum_ki(self.J, Y, index, 'z+-')-sum_li(self.J, Y, index, 'z-+'))\
                       -0.5*(self.gamma+self.gamma.T)*Y[index['-+']]


        D[index['++']]=1j*(self.eps+self.eps.T)*Y[index['++']]\
                       -2j*(sum_ki(self.J, Y, index, 'z++')+sum_li(self.J, Y, index, 'z++'))\
                       -0.5*(self.gamma+self.gamma.T)*Y[index['++']] 

        D[index['--']]=-1j*(self.eps+self.eps.T)*Y[index['--']]\
                        +2j*(sum_ki(self.J, Y, index, 'z--')+sum_li(self.J, Y, index, 'z--'))\
                        -0.5*(self.gamma+self.gamma.T)*Y[index['--']]  
 
                        
        D[index['zz']]=1j*(sum_ki(self.J, Y, index, '-z+')-sum_ki(self.J, Y, index, '+z-'))\
                       +1j*(sum_li(self.J, Y, index, '-z+')-sum_li(self.J, Y, index, '+z-'))\
                       -(self.gamma+self.gamma.T)*Y[index['zz']]-0.5*self.gamma*Y[index['z']].T-0.5*self.gamma.T*Y[index['z']]
                       
                       
        D[index['+z']]=1j*self.eps*Y[index['+z']]-0.5j*self.J*Y[index['+']].T\
                       -2j*sum_ki(self.J,Y,index,'zz+')+1j*(sum_li(self.J, Y, index, '-++')-sum_li(self.J, Y, index, '++-'))\
                       -(0.5*self.gamma+self.gamma.T)*Y[index['+z']]-0.5*Y[index['+']]*self.gamma.T
                       
        D[index['-z']]=-1j*self.eps*Y[index['-z']]+0.5j*self.J*Y[index['-']].T\
                        +2j*sum_ki(self.J,Y,index,'zz-')+1j*(sum_li(self.J, Y, index, '--+')-sum_li(self.J, Y, index, '+--'))\
                        -(0.5*self.gamma+self.gamma.T)*Y[index['-z']]-0.5*Y[index['-']]*self.gamma.T   
                        

                       
        D[index['z+']]=1j*self.eps.T*Y[index['z+']]-0.5j*self.J*Y[index['+']]\
                       -2j*sum_li(self.J,Y,index,'zz+')+1j*(sum_ki(self.J, Y, index, '-++')-sum_ki(self.J, Y, index, '++-'))\
                       -(0.5*self.gamma.T+self.gamma)*Y[index['z+']]-0.5*Y[index['+']].T*self.gamma                       

        D[index['z-']]=-1j*self.eps.T*Y[index['z-']]+0.5j*self.J*Y[index['-']]\
                       +2j*sum_li(self.J,Y,index,'zz-')+1j*(sum_ki(self.J, Y, index, '--+')-sum_ki(self.J, Y, index, '+--'))\
                       -(0.5*self.gamma.T+self.gamma)*Y[index['z-']]-0.5*Y[index['-']].T*self.gamma 

                         
                        
                        
                        

        # D[index['--']]=D[index['++']].conjugate()
        # D[index['-z']]=D[index['+z']].conjugate()

                       
        #D[index['-+']]=D[index['+-']].conjugate()
        # D[index['z+']]=D[index['+z']].T  
        # D[index['z-']]=D[index['-z']].T 

        # D[index['zz']]=np.real(D[index['zz']])
        # D[index['+-']]=((D[index['+-']].real+D[index['+-']].T.real)/2+1j*(D[index['+-']].imag-D[index['+-']].T.imag)/2)
        # D[index['-+']]=D[index['+-']].T
        
        for key in index:
            if len(key)>1:
                D[np.diag(index[key])]=0           
      
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
    
    
    
def sum1(J,Y,index,pmz):  

    """
    1st order double operator sum
    pmz='+-'  
   (k,i) the second is free index !
    
   """ 
    vec1=Y[index[pmz[0]]]
    vec2=Y[index[pmz[1]]]
    
    s=vec1*J@vec2  # element wise and matrix product    
    return s       # keep (N,1) dimension

def sum2(J,Y,index,pmz): # 'pmz'='+-'

    """
    2nd order double operator sum
    pmz='+-'  
   (k,i) the second is free index !
    """
    mat=Y[index[pmz]] 
    s=np.sum(J*mat,axis=1,keepdims=True)
    
    return s 
    
def sum2_loop(J,Y,index,pmz):
    
    mat=Y[index[pmz]]
    N=mat.shape[0]
    s=np.zeros((N,1))
    
    for k in range(N):
        for i in range(N):
            s[k] += J[k,i]*mat[k,i]
            
    return s       
    


    
def sum_ki(J,Y, index, pmz): 
    
    """
    2nd order triple operator sum
    pmz='+-z'   
    (k,l,i) assume the third index is always the free index i, sum over i 
    also assume the first index k goes into the the coefficient J_ki 
    """

    vec1=Y[index[pmz[0]]]
    vec2=Y[index[pmz[1]]]
    vec3=Y[index[pmz[2]]] #single (N,1) vectors 
    

    
    mat12=Y[index[(pmz[0]+pmz[1])]]
    mat13=Y[index[(pmz[0]+pmz[2])]]
    mat23=Y[index[(pmz[1]+pmz[2])]] #double operator matrix
    
     
    # s_kl_i=mat12*(J@vec3)
    # s_ki_l=np.sum(mat13*J, axis=1,keepdims=True)*vec2.T
    # s_li_k=J@mat23.T*vec1
    # s_kli=(J@vec3)*(vec1*vec2.T)

    s_kl_i=mat12*(J@vec3)-J*mat12*vec3.T
    s_ki_l=np.sum(mat13*J, axis=1,keepdims=True)*vec2.T-J*mat13*vec2.T
    s_li_k=J@mat23.T*vec1-J*(np.expand_dims(np.diag(mat23),0))*vec1  #expand the dialgonal element to (1,N) row before boradcasting
    s_kli=(J@vec3)*(vec1*vec2.T)-J*(vec1*vec2.T)*vec3.T 
    
    return s_kl_i+s_ki_l+s_li_k-2*s_kli
#    return s_kl_i, s_ki_l,s_li_k, s_kli   
#    return s_kli
   
def sum_li(J,Y, index, pmz):

    r=sum_ki(J, Y, index, pmz)
    
    return r.T
    

def sum_ki_loop(J,Y,index,pmz):
    
    # pmz order is (k,l,i)
    
    vec1=Y[index[pmz[0]]]
    vec2=Y[index[pmz[1]]]
    vec3=Y[index[pmz[2]]] #single (N,1) vectors 
    
    
    mat12=Y[index[(pmz[0]+pmz[1])]]
    mat13=Y[index[(pmz[0]+pmz[2])]]
    mat23=Y[index[(pmz[1]+pmz[2])]] #double operator matrix

    N=mat12.shape[0]    
    
    s1=np.zeros((N,N))
    s2=np.zeros((N,N))
    s3=np.zeros((N,N)) 
    s4=np.zeros((N,N))
    
    for k in range(N):
        for l in range(N):
            for i in range(N):
                if i==k or i==l:
                    continue 
                s1[k,l] += J[k,i]*mat12[k,l]*vec3[i][0]
                s2[k,l] += J[k,i]*mat13[k,i]*vec2[l][0]
                s3[k,l] += J[k,i]*mat23[l,i]*vec1[k][0]
                s4[k,l] += J[k,i]*vec1[k][0]*vec2[l][0]*vec3[i][0]
                
    s=s1+s2+s3-2*s4            
                
 #   return s1,s2,s3,s4       
    return s                       

        
def sum_li_loop(J,Y,index,pmz):

    # pmz order is (l,k,i)
    
    vec1=Y[index[pmz[0]]]
    vec2=Y[index[pmz[1]]]
    vec3=Y[index[pmz[2]]] #single (N,1) vectors 
    
    
    mat12=Y[index[(pmz[0]+pmz[1])]]
    mat13=Y[index[(pmz[0]+pmz[2])]]
    mat23=Y[index[(pmz[1]+pmz[2])]] #double operator matrix

    N=mat12.shape[0]    

    s1=np.zeros((N,N))
    s2=np.zeros((N,N))
    s3=np.zeros((N,N)) 
    s4=np.zeros((N,N))
    
    for k in range(N):
        for l in range(N):
            for i in range(N):
                if i==k or i==l:
                    continue 
                s1[k,l] += J[l,i]*mat12[l,k]*vec3[i][0]
                s2[k,l] += J[l,i]*mat13[l,i]*vec2[k][0]
                s3[k,l] += J[l,i]*mat23[k,i]*vec1[l][0]
                s4[k,l] += J[l,i]*vec1[l][0]*vec2[k][0]*vec3[i][0]
    s=s1+s2+s3-2*s4            
                
#    return s1,s2,s3,s4       
    return s    
    
    
    
    