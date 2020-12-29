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


last_epochs = 30


# In[3]:


from progressbar import progressbar

from gas_example.optimization.vf import create_vfs_time_list, update_vf_models
from gas_example.setup import TIME_EPOCHS
from gas_example.simulation.state import State
from gas_example.enum_types import PowerplantState


from gas_example.optimization.optimization import get_best_action
from gas_example.simulation.simulation import balance_to_pce
from gas_example.enum_types import Action
from gas_example.setup import BORROW_RATE_EPOCH, RISK_FREE_RATE_EPOCH

import numpy as np
import matplotlib.pyplot as plt

def adp_algorithm_complete():
    vfs = create_vfs_time_list()

    for time_epoch_left in progressbar(range(last_epochs)):
        time_epoch = TIME_EPOCHS - time_epoch_left-1
        # Last Vf value is zero, no money to be gained.
        if time_epoch != TIME_EPOCHS - 1:
            update_vf_models(vfs[time_epoch], vfs[time_epoch + 1])
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


from gas_example.optimization.optimization import get_best_action, get_utility_realization
from gas_example.optimization.basis_function import uf_2_inv
init_state=State(10, 25, 55, PowerplantState.NOT_BUILT, 0)


# In[6]:


vfs_0 = vfs[TIME_EPOCHS - last_epochs]
vfs_1 = vfs[TIME_EPOCHS - last_epochs+1]
vfs_2 = vfs[TIME_EPOCHS-1]


# In[7]:


expected_utility = vfs_0.compute_value(init_state)
expected_utility


# In[8]:


result_1 = uf_2_inv(expected_utility)/1_000_000
result_1


# In[9]:


def balance_to_pce(balance):
    if balance > 0:
        pce = balance / (RISK_FREE_RATE_EPOCH ** (last_epochs-1))
    else:
        pce = balance / (BORROW_RATE_EPOCH ** (last_epochs-1))

    return pce


# 239M 

# Thus we expect to gain 240 milions, when deciding based on these Vfs from the initial state stated above

# In[10]:


from gas_example.optimization.optimization import get_best_action, pce


# In[19]:


final_pces = []
for i in progressbar(range(1000)):
    state = init_state
    fcfs = []
    for epoch in range(last_epochs-1):
        action, exp_utility_this_action = get_best_action(state, vfs[TIME_EPOCHS-last_epochs+epoch+1])
        state, fcf = state.get_new_state_and_fcf(action)
        fcfs.append(fcf)
    final_pces.append(round(balance_to_pce(state.balance) / 1000000.0, 5))
    


# In[20]:


plt.hist(final_pces)


# In[21]:


result_2 = np.mean(final_pces)
result_2


# In[22]:


result_2 - result_1


# In[23]:


(result_2 - result_1)/result_2*100


# 13% 

# In[16]:


state = init_state


# In[17]:


state = init_state
fcfs = []
for epoch in range(last_epochs-1):
    action, exp_utility_this_action = get_best_action(state, vfs[TIME_EPOCHS-last_epochs+epoch+1], True)
    print(action)
    print(exp_utility_this_action)
    state, fcf = state.get_new_state_and_fcf(action)
    print(state.to_dict())
    fcfs.append(fcf)
    print(f"Fcf: {fcf}")
    print(f"Balance {state.balance}")
    


# In[18]:


state_1.get_new_state_and_fcf(Action.RUN)


# In[ ]:


uf_2_inv(vfs_0.compute_value(init_state))


# In[ ]:


get_best_action(init_state, vfs_1, print_details=True)


# In[ ]:


get_utility_realization(init_state, Action.IDLE_AND_BUILD, vfs_1, print_details=True)


# In[ ]:


state_1, fcf = init_state.get_new_state_and_fcf(Action.IDLE_AND_BUILD)


# In[ ]:


get_utility_realization(state_1, Action.RUN, vfs_2, print_details=True)


# In[ ]:





# In[ ]:


df_vfs = pd.DataFrame(vfs)
identificator = pd.Timestamp.now().strftime("%Y-%m-%d_H%H")
df_vfs.to_pickle(f'saved_vfs/vfs_{identificator}.pkl')


# In[ ]:


df_vfs_2 = pd.read_pickle(f'saved_vfs/vfs_{identificator}.pkl')
vfs_2 = list(df_vfs_2[0])


# In[ ]:





# In[ ]:


from gas_example.enum_types import PowerplantState, Action


# In[ ]:





# In[ ]:


new_state.to_dict()

