# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 14:49:57 2021

@author: User
"""


import numpy as np

    
def random_uniform(num_sites, W):
    energies=W*(np.random.random([num_sites])-0.5) #sample uniformly from [-W/2, W/2)
    return energies 


def sum_onsite(energies):
    return energies*2-np.sum(energies)













# from numpy.random import default_rng
# import config
# from energyComputers.energyComputer import EnergyComputer

# class UniformRandomEnergies(EnergyComputer):

#   def __init__(self, W, num_sites):
#     super().__init__(num_sites)
#     self.W = W  # The default unit for W is GHz
#     self.desc = "Onsite energies chosen from a uniform random distribution between -W/2 and W/2.\n"
#     self.rng = default_rng(config.SEED)

#   def get_energies(self):
#     self.energies = self.rng.uniform(-self.W/2, self.W/2, self.num_sites)
#     return self.energies
