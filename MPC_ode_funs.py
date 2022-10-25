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
        
        for k in range(self.N):
            D[index['z'][k]]=0
            D[index['+'][k]]=1j*self.eps[k]*Y[index['+'][k]]
            D[index['-'][k]]=-1j*self.eps[k]*Y[index['-'][k]] 
            for i in range(self.N):
               if i==k: continue 
               D[index['z'][k]] += 1j*self.J[i,k]*(Y[index['+'][i]]*Y[index['-'][k]]-Y[index['+'][k]]*Y[index['-'][i]])
               D[index['+'][k]] += -2j*self.J[i,k]*Y[index['+'][i]]*Y[index['z'][k]]   
               D[index['-'][k]] += +2j*self.J[i,k]*Y[index['-'][i]]*Y[index['z'][k]]

               
            #dissipation term         
            D[index['z'][k]] += -self.gamma[k]*(Y[index['z'][k]]+0.5)
            D[index['+'][k]] += -self.gamma[k]*Y[index['+'][k]]/2
            D[index['-'][k]] += -self.gamma[k]*Y[index['-'][k]]/2    
        return D    
    


# =============================================================================
#     def fun_2nd_all(self, t, Y, index):
# =============================================================================
        """
         by 2nd order approximation:
             <abc>=<ab><c>+<ac><b>+<bc><a>-2<a><b><c>  
             
        The index thus contain 'z', '+', '-';  '+-', '++', '--', '+z', '-z', 'zz'     
    
            
        no order swapping
        """
        
        # last_t, dt = state
        # n = int((t - last_t)/dt)
        # pbar.update(n)
        # state[0] = last_t + dt * n        
        
        D=0*Y
        
        for k in range(self.N):
            D[index['z'][k]]=0   #for Sz
            D[index['+'][k]]=1j*self.eps[k]*Y[index['+'][k]]  # for Sp
            
            for i in range(self.N):
                if i==k: continue
                D[index['z'][k]] += 1j*self.J[i,k]*(Y[index['+-'][i,k]]-Y[index['+-'][k,i]])
                                         
                D[index['+'][k]] += -2j*self.J[i,k]*Y[index['+z'][i,k]]
                
            if self.Diss=='dephasing':
                pass
                # D[index['z'][k]] +=0                   
                # D[index['+'][k]] +=-1/2*self.gamma[k]*Y[index['+'][k]]                  
                    
            elif self.Diss=='dissipation':
                D[index['z'][k]] += -self.gamma[k]*(Y[index['z'][k]]+0.5)
                D[index['+'][k]] += -0.5*self.gamma[k]*Y[index['+'][k]]
            else:
                print('Jump operator is not difined')   
                    
            D[index['-'][k]] = D[index['+'][k]].conjugate()
            
            for l in range(self.N):
                if l==k: continue
                D[index['+-'][k,l]]=1j*(self.eps[k]-self.eps[l])*Y[index['+-'][k,l]]-1j*self.J[k,l]*(Y[index['z'][k]]-Y[index['z'][l]]) # for SpSm
                
                D[index['++'][k,l]]=1j*(self.eps[k]+self.eps[l])*Y[index['++'][k,l]]  # for SpSp
                
               # D[index['-+'][k,l]]=1j*(self.eps[k]-self.eps[l])*Y[index['-+'][k,l]] #for SmSp
                
                D[index['+z'][k,l]]=1j*self.eps[k]*Y[index['+z'][k,l]]-0.5j*self.J[k,l]*Y[index['+'][l]] # for SpSz
                
                D[index['-z'][k,l]]=-1j*self.eps[k]*Y[index['-z'][k,l]]+0.5j*self.J[k,l]*Y[index['-'][l]] # for SmSz
                
                D[index['zz'][k,l]]=0 # for SzSz
                
                for i in range(self.N):
                    if i==l or i==k: continue
                    D[index['+-'][k,l]] += -2j*(self.J[k,i]*self.SSS('+-z',[i,l,k],Y,index)\
                                               -self.J[l,i]*self.SSS('+-z',[k,i,l],Y,index))
                        
                    
                    D[index['++'][k,l]] += -2j*(self.J[k,i]*self.SSS('++z',[i,l,k],Y,index)\
                                               +self.J[l,i]*self.SSS('++z',[k,i,l],Y,index))
                   
                           
                           
                    D[index['+z'][k,l]] += 1j*self.J[l,i]*(self.SSS('++-',[i,k,l],Y,index)-self.SSS('++-',[k,l,i],Y,index))\
                                           -2j*self.J[k,i]*self.SSS('+zz',[i,k,l],Y,index)

                    D[index['-z'][k,l]] += 1j*self.J[l,i]*(self.SSS('+--',[i,k,l],Y,index)-self.SSS('+--',[l,k,i],Y,index))\
                                           +2j*self.J[k,i]*self.SSS('-zz',[i,k,l],Y,index)
                                
                     
                    D[index['zz'][k,l]] += 1j*self.J[k,i]*(self.SSS('+-z',[i,k,l],Y,index)-self.SSS('+-z',[k,i,l],Y,index))\
                                          +1j*self.J[l,i]*(self.SSS('+-z',[i,l,k],Y,index)-self.SSS('+-z',[l,i,k],Y,index))
                        
                            
                        
                           
                if self.Diss=='dephasing':          
                    pass
                    # D[index['+-'][k,l]] += 1/2*(self.gamma[k]+self.gamma[l])*((l==k)-1)*Y[index['+-'][k,l]]
                    
                    # D[index['++'][k,l]] +=-1/2*(self.gamma[k]+self.gamma[l])*((l==k)+1)*Y[index['++'][k,l]]
                    
                    # D[index['-+'][k,l]] += 1/2*(self.gamma[k]+self.gamma[l])*((l==k)-1)*Y[index['-+'][k,l]]
                    
                    # D[index['+z'][k,l]] +=-1/2*self.gamma[k]*Y[index['+z'][k,l]]
                    
                    # D[index['-z'][k,l]] +=-1/2*self.gamma[k]*Y[index['-z'][k,l]]
                    
                    # D[index['zz'][k,l]] +=0
                    
                elif self.Diss=='dissipation':
                    D[index['+-'][k,l]] += -0.5*(self.gamma[k]+self.gamma[l])*Y[index['+-'][k,l]]
                
                    D[index['++'][k,l]] += -0.5*(self.gamma[k]+self.gamma[l])*Y[index['++'][k,l]]
                    
                    
                    D[index['+z'][k,l]] += -(0.5*self.gamma[k]+self.gamma[l])*Y[index['+z'][k,l]]-0.5*self.gamma[l]*Y[index['+'][k]]
                        
                    D[index['-z'][k,l]] += -(0.5*self.gamma[k]+self.gamma[l])*Y[index['-z'][k,l]]-0.5*self.gamma[l]*Y[index['-'][k]]
                    
                    D[index['zz'][k,l]] += -(self.gamma[k]+self.gamma[l])*Y[index['zz'][k,l]]\
                                           -0.5*self.gamma[k]*Y[index['z'][l]]-0.5*self.gamma[l]*Y[index['z'][k]]

                                              
                else:
                    print('Jump operator is not difined')
                    
                    
                    
                D[index['--'][l,k]]=np.conjugate(D[index['++'][k,l]])
                # D[index['z-'][l,k]]=np.conjugate(D[index['+z'][k,l]])
                # D[index['z+'][l,k]]=np.conjugate(D[index['-z'][k,l]])
                
                for key in index:
                    if len(key)>1:
                        D[np.diag(index[key])]=0           
              
                return D

                          
        return D    
            

    def fun_2nd_new(self, t, Y, index):
        """
         by 2nd order approximation:
             <abc>=<ab><c>+<ac><b>+<bc><a>-2<a><b><c>  
             
        The index thus contain 'z', '+', '-';  '+-', '++', '--', '+z', '-z', 'zz'     
    
        
        """
        
        # last_t, dt = state
        # n = int((t - last_t)/dt)
        # pbar.update(n)
        # state[0] = last_t + dt * n        
        
        D=0*Y
        
        for k in range(self.N):
            D[index['z'][k]]=0   #for Sz
            D[index['+'][k]]=1j*self.eps[k]*Y[index['+'][k]]  # for Sp
            D[index['-'][k]]=-1j*self.eps[k]*Y[index['-'][k]] # for Sm
            for i in range(self.N):
                if i==k: continue
                # D[index['z'][k]] += 1j*self.J[i,k]*(Y[index['+-'][i,k]]-Y[index['+-'][k,i]])
                                         
                # D[index['+'][k]] += -2j*self.J[i,k]*Y[index['+z'][i,k]]
                
                # D[index['-'][k]] += +2j*self.J[i,k]*Y[index['-z'][i,k]]
                                
                
                D[index['z'][k]] += 1j*self.J[i,k]*(self.SS('+-',[i,k],Y,index)-self.SS('+-',[k,i],Y,index))
                                         
                D[index['+'][k]] += -2j*self.J[i,k]*self.SS('+z',[i,k],Y,index)
                
                D[index['-'][k]] += +2j*self.J[i,k]*self.SS('-z',[i,k],Y,index)              
                
          
                
            if self.Diss=='dephasing':
                pass             
                    
            elif self.Diss=='dissipation':
                D[index['z'][k]] += -self.gamma[k]*(Y[index['z'][k]]+0.5)
                D[index['+'][k]] += -0.5*self.gamma[k]*Y[index['+'][k]]
                D[index['-'][k]] += -0.5*self.gamma[k]*Y[index['-'][k]]
            else:
                print('Jump operator is not difined')   
                    
#            D[index['-'][k]] = D[index['+'][k]].conjugate()
            
            
#        for k in range(self.N): 
            for l in range(self.N):
                if l==k: 
                    continue
            
                D[index['+-'][k,l]]= 1j*(self.eps[k]-self.eps[l])*Y[index['+-'][k,l]]-1j*self.J[k,l]*(Y[index['z'][k]]-Y[index['z'][l]]) # for SpSm
                
                D[index['++'][k,l]]= 1j*(self.eps[k]+self.eps[l])*Y[index['++'][k,l]]  # for SpSp
                
                D[index['--'][k,l]]= -1j*(self.eps[k]+self.eps[l])*Y[index['--'][k,l]]  # for SmSm               
                
                D[index['+z'][k,l]]= 1j*self.eps[k]*Y[index['+z'][k,l]]-0.5j*self.J[k,l]*Y[index['+'][l]] # for SpSz
                
                D[index['-z'][k,l]]= -1j*self.eps[k]*Y[index['-z'][k,l]]+0.5j*self.J[k,l]*Y[index['-'][l]] # for SmSz
                
                D[index['zz'][k,l]]=0 # for SzSz
                
                
                for i in range(self.N):
                    if i==l or i==k:
                        continue
                    D[index['+-'][k,l]] += -2j*(self.J[k,i]*self.SSS('+-z',[i,l,k],Y,index)\
                                               -self.J[l,i]*self.SSS('+-z',[k,i,l],Y,index))
                        
                    
                    D[index['++'][k,l]] += -2j*(self.J[k,i]*self.SSS('++z',[i,l,k],Y,index)\
                                               +self.J[l,i]*self.SSS('++z',[k,i,l],Y,index))

                    D[index['--'][k,l]] += 2j*(self.J[k,i]*self.SSS('--z',[i,l,k],Y,index)\
                                               +self.J[l,i]*self.SSS('--z',[k,i,l],Y,index))                    
                           
                           
                    D[index['+z'][k,l]] += 1j*self.J[l,i]*(self.SSS('++-',[i,k,l],Y,index)-self.SSS('++-',[l,k,i],Y,index))\
                                           -2j*self.J[k,i]*self.SSS('+zz',[i,k,l],Y,index)

                    D[index['-z'][k,l]] += 1j*self.J[l,i]*(self.SSS('+--',[i,k,l],Y,index)-self.SSS('+--',[l,k,i],Y,index))\
                                           +2j*self.J[k,i]*self.SSS('-zz',[i,k,l],Y,index)
                                
                     
                    D[index['zz'][k,l]] += 1j*self.J[k,i]*(self.SSS('+-z',[i,k,l],Y,index)-self.SSS('+-z',[k,i,l],Y,index))\
                                          +1j*self.J[l,i]*(self.SSS('+-z',[i,l,k],Y,index)-self.SSS('+-z',[l,i,k],Y,index))
                        


                            
                # dissipation term
                              
                D[index['+-'][k,l]] += -0.5*(self.gamma[k]+self.gamma[l])*Y[index['+-'][k,l]]
            
                D[index['++'][k,l]] += -0.5*(self.gamma[k]+self.gamma[l])*Y[index['++'][k,l]]
                
                D[index['--'][k,l]] += -0.5*(self.gamma[k]+self.gamma[l])*Y[index['--'][k,l]]                    
                
                D[index['+z'][k,l]] += -(0.5*self.gamma[k]+self.gamma[l])*Y[index['+z'][k,l]]-0.5*self.gamma[l]*Y[index['+'][k]]
                    
                D[index['-z'][k,l]] += -(0.5*self.gamma[k]+self.gamma[l])*Y[index['-z'][k,l]]-0.5*self.gamma[l]*Y[index['-'][k]]
                
                D[index['zz'][k,l]] += -(self.gamma[k]+self.gamma[l])*Y[index['zz'][k,l]]\
                                       -0.5*self.gamma[k]*Y[index['z'][l]]-0.5*self.gamma[l]*Y[index['z'][k]]

               

                                      
        # for k in range(self.N): 
        #     for l in range(self.N):
        #         if l==k: 
        #             continue     
        # D[index['-']]=np.conjugate(D[index['+']])                          
        #D[index['--']]=np.transpose(np.conjugate(D[index['++']]))
                # D[index['z-'][l,k]]=np.conjugate(D[index['+z'][k,l]])
                # D[index['z+'][l,k]]=np.conjugate(D[index['-z'][k,l]])
                
        for key in index:
            if len(key)>1:
                D[np.diag(index[key])]=0           
      
        return D
                
                         
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
        return Y[index[pmz][(lm[0], lm[1])]] # 2nd order
    
        # return Y[index[pmz[0]][lm[0]]]*Y[index[pmz[1]][lm[1]]]  # 1st order

    
    def SSS(self, pmz, lm, Y, index):
        
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
                  -2*Y[index[pmz[0]][lm[0]]]*Y[index[pmz[1]][lm[1]]]*Y[index[pmz[2]][lm[2]]]  #2nd order
                 
        # return Y[index[pmz[0]][lm[0]]]*Y[index[pmz[1]][lm[1]]]*Y[index[pmz[2]][lm[2]]]  # 1st order
    
    
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
    
    
    