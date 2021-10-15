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
    
    def fun_1st(self,t,Y, index, pbar, state):
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
    
    def fun_2nd(self, t, Y, index):
        """
        return derivative of <Sz> and <SpSm> by breaking <SpSmSz=<SpSm><Sz>
        """      
        D=0*Y+0j
        
        for l in range(self.N):
            f1=0
            #f2=1j*self.eps[l]*Y[index['+'][l]]
            for i in range(self.N):
                f1 += 2j*self.J[i,l]*(Y[index['+-'][i,l]]-Y[index['+-'][l,i]])
                                         
                #f2 += -4j*self.J[i,l]*Y[index['+'][i]]*Y[index['z'][l]]\
                #       -2j*self.U[i,l]*Y[index['+'][l]]*Y[index['z'][i]] 
            if self.Diss=='dephasing':
                    g1=0                   
                    #g2=-1/2*self.gamma[l]*Y[index['+'][l]]                  
                    
            elif self.Diss=='dissipation':
                    g1=-self.gamma[l]*Y[index['+-'][l,l]]
                    #g2=self.gamma[l]*Y[index['+'][l]]*Y[index['z'][l]]   
            else:
                    print('Jump operator is not difined')                      
            D[index['z'][l]] = f1+g1
            #D[index['+'][l]] = f2+g2              
            
            for m in range(self.N):
                f3=1j*(self.eps[l]-self.eps[m])*Y[index['+-'][l,m]] 
                
                for i in range(self.N):
                    f3 += -4j*(self.J[l,i]*Y[index['+-'][i,m]]*Y[index['z'][l]]\
                              -self.J[m,i]*Y[index['+-'][l,i]]*Y[index['z'][m]]\
                              -(l==m)*self.J[l,i]*Y[index['+-'][i,l]])\
                       +2j*((self.U[m,i]-self.U[l,i])*Y[index['+-'][l,m]]*Y[index['z'][i]]\
                            +self.U[m,l]*Y[index['+-'][l,m]])                  
                if self.Diss=='dephasing':             
                    g3=1/2*(self.gamma[l]+self.gamma[m])*((m==l)-1)*Y[index['+-'][l,m]]
                    
                elif self.Diss=='dissipation':
                    g3=self.gamma[l]*Y[index['+-'][l,m]]*Y[index['z'][l]]\
                        +self.gamma[m]*Y[index['+-'][l,m]]*Y[index['z'][m]]\
                        -self.gamma[m]*(1+(l==m))*Y[index['+-'][l,m]]    
                else:
                    print('Jump operator is not difined')
                
                D[index['+-'][l,m]]=f3+g3
        return D   


    def fun_2nd_all(self, t, Y, index):
        """
         by 2nd order approximation:
             <abc>=<ab><c>+<ac><b>+<bc><a>-2<a><b><c>  
             
        The index thus contain 'z', '+', '-' '+-', '++', '--', '-+', 'z+', 'z-' , '+z', '-z', 'zz'     
    
        """
        
        # last_t, dt = state
        # n = int((t - last_t)/dt)
        # pbar.update(n)
        # state[0] = last_t + dt * n        
        
        D=0*Y
        
        for l in range(self.N):
            D[index['z'][l]]=0   #for Sz
            D[index['+'][l]]=1j*self.eps[l]*Y[index['+'][l]]  # for Sp
            
            for i in range(self.N):
                D[index['z'][l]] += 2j*self.J[i,l]*(Y[index['+-'][i,l]]-Y[index['+-'][l,i]])
                                         
                D[index['+'][l]] += -4j*self.J[i,l]*Y[index['+z'][i,l]]\
                                    -2j*self.U[i,l]*Y[index['+z'][l,i]] 
                
            if self.Diss=='dephasing':
                D[index['z'][l]] +=0                   
                D[index['+'][l]] +=-1/2*self.gamma[l]*Y[index['+'][l]]                  
                    
            elif self.Diss=='dissipation':
                D[index['z'][l]] += -self.gamma[l]*self.SS('+-',[l,l],Y,index)
                D[index['+'][l]] += self.gamma[l]*self.SS('+z',[l,l],Y,index) 
            else:
                print('Jump operator is not difined')   
                    
            D[index['-'][l]] = D[index['+'][l]].conjugate()
            
            for m in range(self.N):
                D[index['+-'][l,m]]=1j*(self.eps[l]-self.eps[m])*Y[index['+-'][l,m]]  # for SpSm
                
                D[index['++'][l,m]]=1j*(self.eps[l]+self.eps[m])*Y[index['++'][l,m]]  # for SpSp
                
                D[index['-+'][l,m]]=1j*(self.eps[l]-self.eps[m])*Y[index['-+'][l,m]] #for SmSp
                
                D[index['+z'][l,m]]=1j*self.eps[l]*Y[index['+z'][l,m]] # for SpSz
                
                D[index['-z'][l,m]]=-1j*self.eps[l]*Y[index['-z'][l,m]] # for SmSz
                
                D[index['zz'][l,m]]=0 # for SzSz
                
                for i in range(self.N):
                    D[index['+-'][l,m]] += -4j*(self.J[l,i]*self.SSS('+z-',[i,l,m],Y,index)\
                                               -self.J[m,i]*self.SSS('+z-',[l,m,i],Y,index))\
                       +1j*(self.U[m,i]-self.U[l,i])*(self.SSS('+-z',[l,m,i],Y,index)\
                                                      +self.SSS('z+-',[i,l,m],Y,index))\
                          
                    
                    D[index['++'][l,m]] += -4j*(self.J[l,i]*self.SSS('+z+',[i,l,m],Y,index)\
                                               +self.J[m,i]*self.SSS('++z',[l,i,m],Y,index))\
                       -2j*(self.U[m,i]+self.U[l,i])*(self.SSS('++z',[l,m,i],Y,index)\
                                                      +self.SSS('z++',[i,l,m],Y,index))
                    
                    D[index['-+'][l,m]] += +4j*(self.J[l,i]*self.SSS('z+-',[l,m,i],Y,index)\
                                               -self.J[m,i]*self.SSS('+-z',[i,l,m],Y,index))\
                       +1j*(self.U[l,i]-self.U[m,i])*(self.SSS('-+z',[l,m,i],Y,index)\
                                                      +self.SSS('z-+',[i,l,m],Y,index))\
                           
                           
                    D[index['+z'][l,m]] += -2j*self.J[m,i]*(self.SSS('++-',[l,m,i],Y,index)\
                                           -self.SSS('++-',[i,l,m],Y,index))\
                                           -4j*self.J[l,i]*self.SSS('+zz',[i,l,m],Y,index)\
                        -2j*self.U[i,l]*self.SSS('+zz',[l,m,i],Y,index)

                    D[index['-z'][l,m]] += -2j*self.J[m,i]*(self.SSS('-+-',[l,m,i],Y,index)\
                                           -self.SSS('+--',[i,l,m],Y,index))\
                                           +4j*self.J[l,i]*self.SSS('zz-',[l,m,i],Y,index)\
                        +2j*self.U[i,l]*self.SSS('-zz',[l,m,i],Y,index)
                                
                     
                    D[index['zz'][l,m]] += -2j*self.J[m,i]*(self.SSS('z+-',[l,m,i],Y,index)\
                                                            -self.SSS('+z-',[i,l,m],Y,index))\
                                           -2j*self.J[l,i]*(self.SSS('+z-',[l,m,i],Y,index)\
                                                            -self.SSS('+-z',[i,l,m],Y,index))   
                            
                        
                           
                if self.Diss=='dephasing':             
                    D[index['+-'][l,m]] += 1/2*(self.gamma[l]+self.gamma[m])*((m==l)-1)*Y[index['+-'][l,m]]
                    
                    D[index['++'][l,m]] +=-1/2*(self.gamma[l]+self.gamma[m])*((m==l)+1)*Y[index['++'][l,m]]
                    
                    D[index['-+'][l,m]] += 1/2*(self.gamma[l]+self.gamma[m])*((m==l)-1)*Y[index['-+'][l,m]]
                    
                    D[index['+z'][l,m]] +=-1/2*self.gamma[l]*Y[index['+z'][l,m]]
                    
                    D[index['-z'][l,m]] +=-1/2*self.gamma[l]*Y[index['-z'][l,m]]
                    
                    D[index['zz'][l,m]] +=0
                    
                elif self.Diss=='dissipation':
                    D[index['+-'][l,m]] += self.gamma[l]*self.SSS('+z-',[l,l,m],Y,index)\
                                          +self.gamma[m]*self.SSS('+z-',[l,m,m],Y,index)
                
                    D[index['++'][l,m]] += self.gamma[l]*self.SSS('+z+',[l,l,m],Y,index)\
                                         +self.gamma[m]*self.SSS('++z',[l,m,m],Y,index)
                    
                    D[index['-+'][l,m]] += self.gamma[l]*self.SSS('-z+',[l,l,m],Y,index)\
                                          +self.gamma[m]*self.SSS('-z+',[l,m,m],Y,index)
                    
                    D[index['+z'][l,m]] += self.gamma[l]*self.SSS('+zz',[l,l,m],Y,index)\
                                         -self.gamma[m]*self.SSS('++-',[l,m,m],Y,index)
                        
                    D[index['-z'][l,m]] += self.gamma[l]*self.SSS('z-z',[l,l,m],Y,index)\
                                         -self.gamma[m]*self.SSS('+--',[m,l,m],Y,index)
                    
                    D[index['zz'][l,m]] += -self.gamma[l]*self.SSS('+-z',[l,l,m],Y,index)\
                                           -self.gamma[m]*self.SSS('+z-',[m,l,m],Y,index)\

                                              
                else:
                    print('Jump operator is not difined')
                    
                    
                    
                D[index['--'][m,l]]=np.conjugate(D[index['++'][l,m]])
                D[index['z-'][m,l]]=np.conjugate(D[index['+z'][l,m]])
                D[index['z+'][m,l]]=np.conjugate(D[index['-z'][l,m]])

                          
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
            pre += N
            index[key]=value
            
        while double_ops:
            key=double_ops.pop(0)
            #print(pre)
            value=np.array([range(N*i+pre, N*(i+1)+pre)  for i in range(N)])
            pre += N**2
            index[key]=value
            self.index=index
                  
        return index   
    
    
    def SS(self, pmz, lm, Y, index): 
        """
        length 2
        
        e.g. pmz='zz', lm=(0,0) and       
        return S_z[0]*S_z[0] 
        """
#        return Y[index[pmz][lm]]
        return Y[index[pmz[0]][lm[0]]]*Y[index[pmz[1]][lm[1]]]

    
    def SSS(self, pmz, lm, Y, index):
        
        """
        length 3
        2nd order approximation:
             <abc>=<ab><c>+<ac><b>+<bc><a>-2<a><b><c>
        
         e.g. pmz='zzz', lm=(0,0,0) and       
        return S_zS_zS_z[0,0,0] 
        """
        # return Y[index[(pmz[0]+pmz[1])][(lm[0], lm[1])]]*Y[index[pmz[2]][lm[2]]]\
        #           +Y[index[(pmz[0]+pmz[2])][(lm[0], lm[2])]]*Y[index[pmz[1]][lm[1]]]\
        #           +Y[index[(pmz[1]+pmz[2])][(lm[1], lm[2])]]*Y[index[pmz[0]][lm[0]]]\
        #           -2*Y[index[pmz[0]][lm[0]]]*Y[index[pmz[1]][lm[1]]]*Y[index[pmz[2]][lm[2]]]
            
        return Y[index[pmz[0]][lm[0]]]*Y[index[pmz[1]][lm[1]]]*Y[index[pmz[2]][lm[2]]]
    
    
    def SSS2(self, pmz, lm, Y, index):
        
        """
        length 3
        2nd order approximation:
             <abc>=<ab><c>+<ac><b>+<bc><a>-2<a><b><c>
        
         e.g. pmz='zzz', lm=(0,0,0) and       
        return S_zS_zS_z[0,0,0] 
        """
        return Y[index[(pmz[0]+pmz[1])][(lm[0], lm[1])]]*Y[index[pmz[2]][lm[2]]]\
                  +Y[index[(pmz[0]+pmz[2])][(lm[0], lm[2])]]*Y[index[pmz[1]][lm[1]]]\
                  +Y[index[(pmz[1]+pmz[2])][(lm[1], lm[2])]]*Y[index[pmz[0]][lm[0]]]\
                  -2*Y[index[pmz[0]][lm[0]]]*Y[index[pmz[1]][lm[1]]]*Y[index[pmz[2]][lm[2]]]
            
        #return Y[index[pmz[0]][lm[0]]]*Y[index[pmz[1]][lm[1]]]*Y[index[pmz[2]][lm[2]]]    
    
    
    