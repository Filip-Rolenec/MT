#!/usr/bin/env python
# coding: utf-8

# # Gas powerplant - ADP investigation 
# - I found out that it is not that easy to make the model of value function. 
# - I have tried to come up with some basis functions that would make sense, but then it didnt. 
# - I have simplified the model to only three price values and one plant state. 
# 
# - In this notebook I want to come up with a better basis functions, that will describe better the reeality. 
# - I will start by precise modelling of the last Vf, then I will go in other time epochs. 

# # 1. Epoch 298 
# - Get a lot of (state, utility) pairs, so that we see the function "plotted" by scatter. Then try to make a good model out of it

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


# In[3]:


sns.set()


# In[4]:


time_epoch = 298
VF_SAMPLE_GLOBAL = 1000
VF_SAMPLE_IND = 100
next_vf = Vf()


# In[5]:


sampled_states = get_state_sample(VF_SAMPLE_GLOBAL, VF_SAMPLE_IND, time_epoch)


# In[6]:


state_reward_pairs_raw = get_state_utility_pairs(sampled_states, time_epoch, next_vf)


# In[ ]:





# In[7]:


def plot_spark_utility_graph(plant_state): 
    pair_choice  = [pair for pair in state_reward_pairs_raw if pair[0].plant_state ==plant_state]
    sparks = [pair[0].get_spark_price() for pair in pair_choice]
    utilities = [pair[1] for pair in pair_choice]
    
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


pair_choice  = [pair for pair in state_reward_pairs_raw if pair[0].plant_state ==plant_state]
sparks = [pair[0].get_spark_price() for pair in pair_choice]
utilities = [pair[1] for pair in pair_choice]


# In[11]:


x = np.array(sparks).reshape(-1,1)
y = np.array(utilities)


# In[12]:


model = LinearRegression().fit(x, y)


# In[13]:


model.coef_


# In[14]:


sparks_sample = np.linspace(-200,200,401)


# In[15]:


a = model.predict(sparks_sample.reshape(-1,1))


# In[16]:


plt.scatter(sparks, utilities)
plt.plot(sparks_sample, a)
plt.xlabel("Spark")
plt.ylabel("Utility")
plt.title(f"Spark vs Utility {plant_state}")
plt.show()


# In[17]:


from scipy import optimize
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


def piecewise_linear(x, x0, y0, k1, k2):
    return np.piecewise(x, [x < x0], [lambda x:k1*x + y0-k1*x0, lambda x:k2*x + y0-k2*x0])

p , e = optimize.curve_fit(piecewise_linear, sparks, utilities)
xd = np.linspace(min(sparks), max(sparks), 500)
plt.plot(sparks, utilities, "o")
plt.plot(xd, piecewise_linear(xd, *p))


# # Looking good 
# - What is the equation? How to get the prediction? By saving a model itself? 
# 

# # Testing

# In[18]:


from gas_example.optimization_2.adp_algorithm import adp_algorithm_complete


# In[19]:


vfs = adp_algorithm_complete()


# In[27]:


p = vfs[0].pw_params[0]
plt.plot(xd, piecewise_linear(xd, *p))


# In[20]:


from gas_example.simulation.state import State
from gas_example.optimization_2.optimization import get_best_action
from gas_example.optimization_2.vf import Vf


# In[21]:


s= State(10,25,44, PowerplantState.NOT_BUILT, 110000000)


# In[22]:


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
