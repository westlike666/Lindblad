# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:27:10 2021

@author: User
"""

import datetime as datetime
import pickle


def get_run_time():
    currentDT = datetime.datetime.now()
    return "%d-%d-%d-%d_%d_%d" % (currentDT.year, currentDT.month, currentDT.day, currentDT.hour, currentDT.minute, currentDT.second)

def store_vars(path, obj):
    f=open(path+'\\store.p', 'wb')
    pickle.dump(obj, f)
    f.close()
    
def load_vars(path):
    f=open(path, 'rb')
    return pickle.load(f)
    
    
    
    


