#!/usr/bin/env python
# coding: utf-8

# # ADP algorithm gas 
# - In this jupyter notebook I will be testing the functions for the final ADP algorithm for my gas powerplant example. 
# - When this file is completed, I will have a function which will be similar to the end of the investigation denoted as adp_algorithm_final. 

# In[2]:


import sys
import os
sys.path.append("/Users/filiprolenec/Desktop/MT/MTpython/src")


# In[ ]:


vfs_1 = adp_algorithm_final(loops_of_update=100, sample_size=2000, problem_setup = problem_setup)


# In[3]:


from gas_example.enum_types import Action


# In[6]:


[action for action in Action]


# In[8]:


[0]*10


# In[ ]:




