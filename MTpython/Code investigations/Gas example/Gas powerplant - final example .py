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


from gas_example.enum_types import MothballedState, PowerplantState
from gas_example.simulation.simulation import run_simulation


# In[3]:


from  gas_example.simulation.state import State, update_balance
import gas_example.simulation.strategy as s
import matplotlib.pyplot as plt
import time
from progressbar import progressbar


# In[4]:


import seaborn as sns
sns.set()


# # Initial settings 

# In[5]:


initial_state = State(24,9,39,1,PowerplantState.NOT_BUILT, MothballedState.NORMAL,0)
epoch = 0 


# In[6]:


strategy_0 = s.Strategy(s.heuristic_strategy_function_0)
strategy_1 = s.Strategy(s.heuristic_strategy_function_1)
strategy_2 = s.Strategy(s.heuristic_strategy_function_2)

strategies = [strategy_0, strategy_1, strategy_2]


# In[7]:


final_balance, balances = run_simulation(strategy_2, initial_state)


# In[8]:


balances


# In[ ]:


import time
import sys

results_final = {}
for i in range(len(strategies)):
    results = []
    for j in progressbar(range(1500)):
        results.append(run_simulation(strategies[i], initial_state))
    results_final[i]= results
    


# In[ ]:


plt.hist(results_final)


# In[ ]:





# In[ ]:





# In[ ]:




