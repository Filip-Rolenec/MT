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
import pandas as pd
import numpy as np
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


final_balance= run_simulation(strategy_2, initial_state)


# In[8]:


import time
import sys

results_final = {}
for i in range(len(strategies)):
    results = []
    for j in progressbar(range(10000)):
        results.append(run_simulation(strategies[i], initial_state))
    results_final[i]= results


# In[9]:


flatten = lambda t: [item for sublist in t.values() for item in sublist]
all_results = flatten(results_final)
max_value_displayed = np.percentile(all_results, 98)
min_value_displayed = np.percentile(all_results, 2)


# In[10]:


total_min = min([min(results_final[i]) for i in range(len(results_final))])
total_min = min_value_displayed
total_max = max_value_displayed

width = (total_max-total_min)/30
b = [total_min +i*width for i in range(30)]


# In[11]:


df = pd.DataFrame(results_final)
means = [np.mean(df[i]) for i in range(len(results_final))]
colors = sns.color_palette()[0:3]


# In[12]:


fig, ax = plt.subplots(figsize = (12,6), dpi = 100)

plt.hist(df, bins = b, label = [f"Strategy_{i}" for i in df.columns])

trans = ax.get_xaxis_transform()


for i,mean in enumerate(means):
    plt.axvline(x=mean,linestyle = "dashed", color = colors[i])
    plt.text(mean+5, 0.5+i*0.05, round(mean),transform = trans,  color = colors[i])
plt.xlabel("M of EUR")
plt.ylabel("Count")
plt.legend()
plt.title("Baseline strategies and their expected PCEs")
plt.show()

