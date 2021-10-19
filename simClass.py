# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 19:14:15 2021

@author: westl
"""

import numpy as np
from XY_class import*
from XY_ode_funs import ode_funs
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time 
from datetime import datetime 
from tqdm import tqdm
#from qutip import*
from energy_paras import Energycomputer, Jcomputer, Ucomputer, Gammacomputer



class simulation():
    def __init__(self, L, N, eps_comp, J_comp, U_comp):
        pass
        
        