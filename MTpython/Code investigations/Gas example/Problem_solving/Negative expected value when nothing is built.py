#!/usr/bin/env python
# coding: utf-8

# # Negative values, when nothing is built. 
# - We have always the option to do nothing. 
# - Why is it then, that the model turns into negative values after some time periods? 

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


# In[ ]:


state = State(300, 200, 200, PowerplantState.NOT_BUILT, 0)


# In[ ]:


vfs[290].compute_value(state)


# In[ ]:




