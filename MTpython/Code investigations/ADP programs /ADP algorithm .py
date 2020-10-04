#!/usr/bin/env python
# coding: utf-8

# # ADP algorihtm 
# - The goal of this file is to first learn how does the ADP work on a simple example and then use it to solve the problem of gas power plant valuation. 
# 
# - First, I will setup a simple example of three states and two actions. Each of the two actions changes the probability distributions of results and "costs" some reward. 
#     - I will compute the optimal strategy for this example with real dynamic programming and then with the approximative dynamic programming. 
#     - I will make heuristic strategies as well. 

# ## Three states two actions

# In[1]:


import sys
import os
sys.path.append("/Users/filiprolenec/Desktop/MT/MTpython/src")


# In[2]:


import simple_example.strategy as s
from simple_example.simulation import run_simulation
import matplotlib.pyplot as plt 
import numpy as np
from simple_example.setup import prob_matrix, reward_matrix


# # Classical Dynamic programming 

# In[3]:


horizon_vf = [0,0,0]
time_epochs = 10 


# In[4]:


def classic_DP(horizon_vf, time_epochs, prob_matrix, reward_matrix): 
    
    time_epochs = 10 
    state_count = len(prob_matrix)
    action_count = len(prob_matrix[0])
    
    vf = {}
    vf_prev_epoch = horizon_vf

    strategy = {} 

    exp_vector = [0,0]

    for epoch in range(time_epochs): 
        vf_epoch = []
        epoch_strategy = {} 
        for state in range(state_count): 
            #print(state)
            for action in range(action_count): 
                exp_vector[action] = sum([i*j for i,j in zip(prob_matrix[state][action], [i+j for i,j in zip(reward_matrix[state][action], vf_prev_epoch)])]) 

            epoch_strategy[state+1] = np.argmax([exp_vector])+1
            vf_epoch.append(np.max([exp_vector]))
        vf_prev_epoch = vf_epoch
        vf[time_epochs - epoch-1] = vf_epoch
        strategy[time_epochs - epoch-1] = epoch_strategy

    return strategy, vf 


# In[5]:


classic_DP(horizon_vf, time_epochs, prob_matrix, reward_matrix)


# # Strategy result comparison

# In[6]:


strategies = [s.heuristic_strategy_0, 
              s.heuristic_strategy_1, 
              s.heuristic_strategy_2, 
              s.optimal_strategy]


# In[7]:


all_results = {}

for strategy in strategies: 
    strategy_results = {}
    for i in range(10000): 
        strategy_results[i] = run_simulation(strategy)
            
    all_results[strategy.__name__] = strategy_results 


# In[8]:


for strategy in strategies: 
    plt.hist(all_results[strategy.__name__].values(), label =strategy.__name__, alpha = 0.7)
plt.legend()


# In[9]:


for strategy in strategies: 
    print(np.mean(list(all_results[strategy.__name__].values())))


# In[ ]:





# In[ ]:





# In[ ]:





# # ADP algorithm finding the best strategy 
# $$V_t(s, \theta) = \theta_0 + \theta_{t,1} \cdot \phi_1(s) + \theta_{t,2} \cdot \phi_2(s) + \theta_{t,3} \cdot \phi_3(s)$$

# where 

# In[ ]:





# In[ ]:




