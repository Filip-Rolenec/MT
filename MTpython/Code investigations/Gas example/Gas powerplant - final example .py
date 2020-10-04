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


from gas_example.enum_types import RunningState, PowerplantState


# In[3]:


from gas_example.simulation import run_simulation


# In[14]:


from  gas_example.state import State
import gas_example.strategy as s
import matplotlib.pyplot as plt
import time
from progressbar import progressbar


# # Initial settings 

# In[15]:


initial_state = State(24,9,39,1,PowerplantState.NOT_BUILT,RunningState.NOT_RUNNING,0)
epoch = 0 


# In[16]:


strategy_0 = s.Strategy(s.heuristic_strategy_0)
strategy_1 = s.Strategy(s.heuristic_strategy_1)
strategies = [strategy_0, strategy_1]


# In[ ]:





# In[17]:


import time
import sys

results_final = {}
for i in range(2):
    results = []
    for j in progressbar(range(1500)):
        results.append(run_simulation(strategies[i], initial_state))
    results_final[i]= results
    


# In[18]:


plt.hist(results_final[0])
plt.hist(results_final[1])


# In[ ]:





# In[ ]:




