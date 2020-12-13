#!/usr/bin/env python
# coding: utf-8

# # ADP simulation 
# - In this notebook we will test the ADP algorithm. 
# 
# - The result of this notebook is a csv that describes dfs parameters in all time intervals, like the example below 

# In[1]:


import sys
import os

sys.path.append("/Users/filiprolenec/Desktop/MT/MTpython/src")


# In[2]:


import pandas as pd
from datetime import datetime
from gas_example.enum_types import MothballedState, PowerplantState
from gas_example.simulation.state import State, update_balance


# In[3]:


from gas_example.enum_types import Action


# In[4]:


df = pd.DataFrame()
df["0"] = [0.3,0.7,13,93,12,-10]
df["1"] = [0.4,0.6,12,10,9,-4]
df.index = ["\u03C6_0",
            "\u03C6_1", 
            "\u03C6_2", 
            "\u03C6_3", 
            "\u03C6_4", 
            "\u03C6_5"]


# In[5]:


df


# In[6]:


import sys
import os

sys.path.append("/Users/filiprolenec/Desktop/MT/MTpython/src")


# In[7]:


from gas_example.optimization.adp_algorithm import adp_algorithm_complete


# In[8]:


vfs = adp_algorithm_complete()


# In[9]:


a = list([10]).append(10)


# In[10]:


a = list([1,2,3])


# In[11]:


a[1:2]


# In[12]:


a = [10]
a.extend([1,2,3])


# In[13]:


a


# In[14]:


(21.2-24.2)*200*720


# In[15]:


param_list = []
for vf in vfs: 
    param_list.append(vf.params)


# In[16]:


df = pd.DataFrame.from_records(param_list).transpose()


# In[18]:


df.index = ["\u03C6_1", 
            "\u03C6_2", 
            "\u03C6_3"]


# In[19]:


pd.set_option('display.max_columns', 300)


# In[20]:


df


# In[ ]:


df.to_csv(f"vfs_1.csv")


# In[ ]:


initial_state = State(24,9,39,1,PowerplantState.NOT_BUILT, MothballedState.NORMAL,0)


# In[ ]:


df_path = pd.read_csv("vfs_1.csv", index_col = [0])


# In[ ]:


df_path


# In[ ]:


list(df_path["0"])


# In[ ]:


Vf()


# In[ ]:


list(10)


# In[ ]:





# In[ ]:





# In[ ]:


get_best_action(state, INTEGRAL_SAMPLE_SIZE, epoch, future_vf)


# In[ ]:




