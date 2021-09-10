# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 18:14:44 2021

@author: User
"""

import numpy as np
from qutip import *
import matplotlib.pyplot as plt



H=Hamiltonian()

rho0=Densitymatrix()

tlist=np.linspace(0.0, 10.0, 200) #(initial,final,devision)

c_op=Jump()

resuts=mesolve(H, rho0, tlist, c_ops=None, e_ops=None, args=None, options=None, progress_bar=None, _safe_mode=True)