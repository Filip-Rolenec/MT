#!/usr/bin/env python
# coding: utf-8

# # ADP simulation 
# - In the previous notebook we have found a better model for the value functions. 
# - Now we want to implement the decision making that is based on such vfs approximations and fix potential bugs. 
# 
# - When approved, this strategy will be run together with the baselines in the next notebook. 

# In[1]:


import sys
import os

sys.path.append("/Users/filiprolenec/Desktop/MT/MTpython/src")

import pandas as pd
from gas_example.optimization_2.vf import Vf
from gas_example.enum_types import PowerplantState
from gas_example.simulation.state import State
from gas_example.optimization_2.optimization import get_best_action

import numpy as np
import matplotlib.pyplot as plt
from gas_example.optimization_2.adp_model import piecewise_linear


# # 1. Decision making based on vf parameters. 

# In[2]:


def get_vfs_from_path(path): 
    df_vfs = pd.read_pickle(path)
    
    vfs = []
    for column in df_vfs: 
        vf = Vf()
        vf.set_params(df_vfs[column])
        vfs.append(vf)
        
    return vfs


# In[10]:


vfs = get_vfs_from_path("saved_vfs/vfs_12_20_2020.pkl")


# In[11]:


init_state=State(10, 25, 44, PowerplantState.NOT_BUILT, 110000000)
time_epoch = 0
get_best_action(init_state, time_epoch,vfs[time_epoch+1])


# ### 20.12.
# - There is a problem with consistency, but it is caused by weird range of sampled values for spark prices. 

# # 2. Run and individual decisions making sense

# In[8]:


vfs = get_vfs_from_path("vfs.csv")


# In[9]:


state = init_state

for i in range(299): 
    action = get_best_action(state, i, vfs[i + 1], True)
    print(action)
    state, fcf = state.get_new_state_and_fcf(action, i)
    print(fcf)


# In[ ]:





# In[ ]:




