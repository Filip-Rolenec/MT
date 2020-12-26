#!/usr/bin/env python
# coding: utf-8

# # Gas powerplant - ADP investigation 
# - I found out that it is not that easy to make the model of value function (Linear model does not work). 
# - I have tried to come up with some basis functions that would make sense, but it seems like the linear model is not capable of capturing the problem. 
# - The complex problem, with government incentive and powerplant months left was simplified. Now the only variables are the prices and the state of a powerplant. Balance might be added to the model as a linear component helping the project.  
# 
# - This notebook is about a better basis functions, or maybe a completely new model of the value function. 
# - I will start by precise modelling of the last Vf, then I will go in other time epochs. 

# # 1. Epoch 298 
# - Get a lot of (state, utility) pairs, so that we see the function "plotted" by scatter. Look at the data and try to moodel them reasonably. 

# In[1]:


import sys
import os
sys.path.append("/Users/filiprolenec/Desktop/MT/MTpython/src")


# In[2]:


from gas_example.optimization.adp_algorithm import adp_algorithm_complete
from gas_example.optimization.optimization import get_state_utility_pairs, get_best_action
from gas_example.optimization.sampling import get_state_sample
from gas_example.optimization.vf import Vf
from gas_example.enum_types import PowerplantState
from gas_example.simulation.state import State


# In[3]:


import matplotlib.pyplot as plt
from scipy import optimize
import seaborn as sns
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
sns.set()


# In[4]:


time_epoch = 298
VF_SAMPLE_GLOBAL = 1000
VF_SAMPLE_IND = 100
next_vf = Vf()


# In[5]:


sampled_states = get_state_sample()
state_reward_pairs_raw = get_state_utility_pairs(sampled_states, next_vf)


# In[6]:


def get_sparks_and_utilities(state_reward_pairs, plant_state): 
    chosen_states  = [state for state in state_reward_pairs_raw.keys() if state.plant_state ==plant_state]
    utilities = [state_reward_pairs_raw[state] for state in state_reward_pairs_raw.keys() if state.plant_state ==plant_state]
    sparks = [state.get_spark_price() for state in chosen_states]
    
    return sparks, utilities


# In[7]:


def plot_spark_utility_graph(plant_state): 
    sparks, utilities = get_sparks_and_utilities(state_reward_pairs_raw, plant_state)
    
    plt.scatter(sparks, utilities)
    plt.xlabel("Spark")
    plt.ylabel("Utility")
    plt.title(f"Spark vs Utility {plant_state}")
    plt.show()


# In[8]:


for state in PowerplantState: 
    plot_spark_utility_graph(state)


# **Stage 1 and 2 are similar, we will not focus on them individually. Trying to find a linear model of stage 1**

# In[9]:


plant_state =PowerplantState.STAGE_1


# In[10]:


sparks, utilities = get_sparks_and_utilities(state_reward_pairs_raw, plant_state)


# In[11]:


x = np.array(sparks).reshape(-1,1)
y = np.array(utilities)
model = LinearRegression().fit(x, y)
model.coef_


# In[12]:


sparks_sample = np.linspace(-200,200,401)
a = model.predict(sparks_sample.reshape(-1,1))


# In[13]:


plt.scatter(sparks, utilities)
plt.plot(sparks_sample, a)
plt.xlabel("Spark")
plt.ylabel("Utility")
plt.title(f"Spark vs Utility {plant_state}")
plt.show()


# # Obsolete, the optimization is not done by curve fit anymore

# In[14]:


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

p , e = optimize.curve_fit(piecewise_linear, sparks, utilities)
xd = np.linspace(min(sparks), max(sparks), 500)
plt.plot(sparks, utilities, "o")
plt.plot(xd, piecewise_linear(xd, *p))


# # Looking good 
# - What is the equation? Now we want to write a strategy that will behave based on these value functions. 
# - We save the vfs parameters as csv and we will use it in another notebook. ADP simulaiton. 
# 

# # Testing
# - All at once, what will be the value function in time 0?

# In[15]:


vfs = adp_algorithm_complete()


# In[ ]:


vfs_models = {}
for i in range(len(vfs)):
    vfs_models[i] = vfs[i].models


# In[ ]:


df_vfs = pd.DataFrame(vfs_params)
identificator = pd.Timestamp.now().strftime("%Y-%m-%d_H%H")
df_vfs.to_pickle(f'saved_vfs/vfs_{identificator}.pkl')


# # Conslusion
# - All seems reasonable, no powerplant has more value, because it is vulnerable for price drops. 
# - The code is now somehow reasonable and ready to be used in the next notebook in simulation.
# 
# - Testing: 
#    1. Look what actions are being performed in what states and if they make sense. 
#    2. Look what is the result, how does it compare to baseline models and to the pce equivalent stated by the Vf. 
#    3. Maybe play more with the constants... 

# # Conclusion and model adjustments 
# - Each night, I will run the model, with visualisation and see, how good the model is. 
# - I will make notes and adjust the model so it is able to catch the pattern better. 
# 
# Individual sample size - 200, global sample size -2000, Integral sample size 100. 
# 

# # 19.12. First run with large sample size 
# - It went good in the first 290 rounds, the option pattern is always there. 
# 
# - Problems: 
#     1. Negative expected price for the first plant state, should be strictly higher than 0, because I can always do. In other model cases, the project can heave negative value, but not here, where we have the option to do nothing. 
#     2. There is s degeneration of the model as we approach time epoch 0. This is caused by low support of the modelled spark prices. We model their variance to be linearly time dependent, but we should put a limit on it. 
#          - I have decided this is not that important... It is an approximation, it can slightly not make sense. 
# - Powerplant cost is very low, 65M now, could be set to 650, or the volatility of prices could be set lower. 

# # 24.12. New format run 
# - I have cleaned up the code and prepared the algorithm for weekly time epochs. 
# - This code will be run through the night of 24.12. 
# - I have the vfs for monthly structure in file 24.12. 

# In[ ]:




