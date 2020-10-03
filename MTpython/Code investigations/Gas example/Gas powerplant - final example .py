#!/usr/bin/env python
# coding: utf-8

# # Gas powerplant 
# - In this notebook I will present the complete solution for the gas power plant valuation problem. 
# - In the end it might be too large and I will need to make helper files for the individual functions. For now I dont know. 

# 

# In[1]:


import sys
import os


sys.path.append("/Users/filiprolenec/Desktop/MT/MTpython/src")


# In[2]:


from enum_types import RunningState, PowerplantState


# In[3]:


from simulation import run_simulation


# In[4]:


from state import State
import strategy as s
import matplotlib.pyplot as plt


# # Initial settings 

# In[5]:


initial_state = State(24,9,39,1,PowerplantState.NOT_BUILT,RunningState.NOT_RUNNING,0)
epoch = 0 


# In[6]:


strategy_0 = s.Strategy(s.heuristic_strategy_0)
strategy_1 = s.Strategy(s.heuristic_strategy_1)
strategies = [strategy_0, strategy_1]


# In[7]:


run_simulation(strategy_1, initial_state)


# In[8]:


results_final = {}
for i in range(2): 
    results = []
    for j in range(1000): 
        print(j)
        results.append(run_simulation(strategies[i], initial_state))
    results_final[i]= results


# In[9]:


results_final[0]


# In[ ]:





# In[10]:


plt.hist(results_final[0])
plt.hist(results_final[1])


# In[11]:


results =[]
for i in range(1000): 
    print(i)
    results.append(run_simulation(strategy_1, initial_state))

