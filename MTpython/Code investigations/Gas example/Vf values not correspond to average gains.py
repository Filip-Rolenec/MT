#!/usr/bin/env python
# coding: utf-8

# # Vf vs average gains. 
# - I have computed the optimal decision making as the parameters of value functions. 
# - According to them I was making decisions, and the average gain was in hundreds of millions. Even thoug majority of time, there was zero gain. 
# 
# - I want to know, how much off is the value function approximation with the actuall average gains and repair this illogial results. Hopefully I will also solve the problem with heuristic strategy being better than the optimal one in the final results. 

# ## Testing idea 
# - Lets make the simulation last only one time epoch.
# - Lets look, how does a non-zero expected value hold with the realizations. Cut the Vf computation when the first large sparks are being profitable, look at what the expected value is and run the simulation what it can be...  

# In[1]:


import sys
import os

sys.path.append("/Users/filiprolenec/Desktop/MT/MTpython/src")


# In[2]:


last_epochs = 300


# In[3]:


from progressbar import progressbar

from gas_example.optimization.vf import create_vfs_time_list, update_vf_models
from gas_example.setup import TIME_EPOCHS
from gas_example.simulation.state import State
from gas_example.enum_types import PowerplantState




def adp_algorithm_complete():
    vfs = create_vfs_time_list()

    for time_epoch_left in progressbar(range(last_epochs)):
        time_epoch = TIME_EPOCHS - time_epoch_left-1
        # Last Vf value is zero, no money to be gained.
        if time_epoch != TIME_EPOCHS - 1:
            update_vf_models(vfs[time_epoch], vfs[time_epoch + 1], time_epoch)
    return vfs


# In[4]:


vfs = adp_algorithm_complete()


# ### VF for only last 5 time epochs 
# - We know the Vf for epoch 300. 
# - We are computing 299,298,297,296 and 295. 
# 
# - In time epoch 295 we look at the expected pce gain by the value function for these 5 epochs. 
# 
# - Then we simulate behaving based on this Vf and record the gains... 

# In[5]:


vfs_0 = vfs[TIME_EPOCHS - last_epochs]


# In[6]:


TIME_EPOCHS - last_epochs


# In[7]:


init_state=State(10, 25, 70, PowerplantState.NOT_BUILT, 0)


# In[8]:


expected_utility = vfs_0.compute_value(init_state)


# In[9]:


expected_utility


# In[10]:


def uf_2_inv(y):
    if y < 0:
        thousands = -((-y) ** 1.2)
    else:
        thousands = y ** 1.25

    return thousands * 1000


# In[11]:


uf_2_inv(expected_utility)/1_000_000


# 239M 

# Thus we expect to gain 240 milions, when deciding based on these Vfs from the initial state stated above

# In[12]:


from gas_example.optimization.optimization import get_best_action
from gas_example.simulation.simulation import balance_to_pce


# In[13]:


final_pces = []
for i in range(300):
    print(i)
    state = init_state
    fcfs = []
    for epoch in range(last_epochs-1):
        action = get_best_action(state, vfs[TIME_EPOCHS-last_epochs+epoch+1])
        state, fcf = state.get_new_state_and_fcf(action)
        fcfs.append(fcf)
    final_pces.append(round(state.balance / 1000000.0, 2))
    


# In[14]:


import numpy as np
import matplotlib.pyplot as plt


# In[15]:


np.mean(final_pces)


# In[16]:


plt.hist(final_pces)


# 

# In[37]:


state = init_state
fcfs = []
for epoch in range(last_epochs-1):
    print(TIME_EPOCHS-last_epochs+epoch)
    action = get_best_action(state, vfs_2[TIME_EPOCHS-last_epochs+epoch+1], print_details = True)
    print(action)
    state, fcf = state.get_new_state_and_fcf(action)
    fcfs.append(fcf)
final_pces.append(round(state.balance / 1000000.0))


# In[20]:


sum(fcfs)/1_000_000


# In[26]:


import pandas as pd


# In[28]:


df_vfs = pd.DataFrame(vfs)
identificator = pd.Timestamp.now().strftime("%Y-%m-%d_H%H")
df_vfs.to_pickle(f'saved_vfs/vfs_{identificator}.pkl')


# In[31]:


df_vfs_2 = pd.read_pickle(f'saved_vfs/vfs_{identificator}.pkl')
vfs_2 = list(df_vfs_2[0])


# In[38]:


f'saved_vfs/vfs_{identificator}.pkl'


# In[ ]:





# In[ ]:





# In[ ]:





# In[40]:


from gas_example.enum_types import PowerplantState, Action


# In[41]:


def get_next_plant_state(state: State, action: Action):
    if action_is_invalid(action, state, print_warning=True):
        raise Exception(f"Action {action} in state {state} is invalid")

    # Building new stage
    if action == Action.IDLE_AND_BUILD or action == Action.RUN_AND_BUILD:
        if state.plant_state == PowerplantState.NOT_BUILT:
            print("Building")
            return PowerplantState.STAGE_1
        elif state.plant_state == PowerplantState.STAGE_1:
            return PowerplantState.STAGE_2
        else:
            raise Exception(f"New stage cannot be built, plant is in a state {state.plant_state}")

    # Other actions do not change the plant state
    else:
        return state.plant_state


# In[45]:


from gas_example.simulation.state import State, action_is_invalid


# In[46]:


init_state.to_dict()


# In[47]:


get_next_plant_state(init_state, Action.IDLE_AND_BUILD)


# In[51]:


new_state, fcf  = init_state.get_new_state_and_fcf( Action.IDLE_AND_BUILD)


# In[52]:


new_state.to_dict()


# In[ ]:




