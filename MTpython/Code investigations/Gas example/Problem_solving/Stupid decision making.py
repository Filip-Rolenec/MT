#!/usr/bin/env python
# coding: utf-8

# # Optimal strategy is dumb 
# - The optimal strategy is making stupid decisions which do not make sense. 
# - Not running the plant when there is spark 30 and doing nothing, why? 

# In[1]:


import sys
import os

sys.path.append("/Users/filiprolenec/Desktop/MT/MTpython/src")


# In[2]:


import pandas as pd
from gas_example.enum_types import PowerplantState
from gas_example.optimization.optimization import get_best_action
from gas_example.simulation.state import State, action_is_invalid


# In[3]:


df_vfs = pd.read_pickle(f'../saved_vfs/vfs_2020-12-25_H09.pkl')
vfs = list(df_vfs[0])


# In[4]:


init_state=State(10, 25, 70, PowerplantState.NOT_BUILT, 0)


# In[5]:


state = init_state
fcfs = []
for epoch in range(300-1):
    print(epoch)
    action, exp_value = get_best_action(state, vfs[epoch+1], print_details = True)
    print(f"Best action is: {action}")
    new_state, fcf = state.get_new_state_and_fcf(action)
    state = new_state


# In[ ]:





# # Conclusion 
# - This problem was caused by imprecise measurement in the computation of the expected utility. Since there is only a small difference in running the plant and not running it, and there is a large variance of results, then it might happen that the expecteed value ends up being illogical

# In[ ]:




