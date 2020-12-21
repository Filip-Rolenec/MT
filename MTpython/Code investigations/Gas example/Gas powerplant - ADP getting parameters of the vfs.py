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


from gas_example.optimization.optimization import get_state_utility_pairs
from gas_example.optimization.sampling import get_state_sample
from gas_example.optimization.vf import Vf
from gas_example.enum_types import PowerplantState

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
sns.set()

from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
from gas_example.optimization_2.adp_algorithm import adp_algorithm_complete

from gas_example.simulation.state import State
from gas_example.optimization_2.optimization import get_best_action
from gas_example.optimization_2.vf import Vf


# In[3]:


time_epoch = 298
VF_SAMPLE_GLOBAL = 1000
VF_SAMPLE_IND = 100
next_vf = Vf()


# In[4]:


sampled_states = get_state_sample(VF_SAMPLE_GLOBAL, VF_SAMPLE_IND, time_epoch)
state_reward_pairs_raw = get_state_utility_pairs(sampled_states, time_epoch, next_vf)


# In[5]:


def plot_spark_utility_graph(plant_state): 
    pair_choice  = [pair for pair in state_reward_pairs_raw if pair[0].plant_state ==plant_state]
    sparks = [pair[0].get_spark_price() for pair in pair_choice]
    utilities = [pair[1] for pair in pair_choice]
    
    plt.scatter(sparks, utilities)
    plt.xlabel("Spark")
    plt.ylabel("Utility")
    plt.title(f"Spark vs Utility {plant_state}")
    plt.show()


# In[6]:


for state in PowerplantState: 
    plot_spark_utility_graph(state)


# **Stage 1 and 2 are similar, we will not focus on them individually. Trying to find a linear model of stage 1**

# In[7]:


plant_state =PowerplantState.STAGE_1


# In[8]:


pair_choice  = [pair for pair in state_reward_pairs_raw if pair[0].plant_state ==plant_state]
sparks = [pair[0].get_spark_price() for pair in pair_choice]
utilities = [pair[1] for pair in pair_choice]


# In[9]:


x = np.array(sparks).reshape(-1,1)
y = np.array(utilities)
model = LinearRegression().fit(x, y)
model.coef_


# In[10]:


sparks_sample = np.linspace(-200,200,401)
a = model.predict(sparks_sample.reshape(-1,1))


# In[11]:


plt.scatter(sparks, utilities)
plt.plot(sparks_sample, a)
plt.xlabel("Spark")
plt.ylabel("Utility")
plt.title(f"Spark vs Utility {plant_state}")
plt.show()


# In[12]:


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

# In[13]:


vfs = adp_algorithm_complete()
p = vfs[0].pw_params[0]
plt.plot(xd, piecewise_linear(xd, *p))


# In[14]:


vfs[0].pw_params

vfs_params = {}
for i in range(len(vfs)):
    vfs_params[i] = vfs[i].pw_params


# In[15]:


df_vfs = pd.DataFrame(vfs_params)


# In[16]:


df_vfs.to_pickle('saved_vfs/vfs_12_21_2020.pkl')


# In[17]:


s= State(10,25,44, PowerplantState.NOT_BUILT, 110000000)
future_vf = Vf()
future_vf.set_params([[0,0,0,100],[0,0,0,0],[0,0,0,0]])


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
