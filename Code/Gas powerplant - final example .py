#!/usr/bin/env python
# coding: utf-8

# # Gas powerplant 
# - In this notebook I will present the complete solution for the gas power plant valuation problem. 
# - In the end it might be too large and I will need to make helper files for the individual functions. For now I dont know. 

# 

# In[80]:


from scipy.stats import norm
import matplotlib.pyplot as plt 
import numpy as np


# In[191]:


def plot_lognorm_price(delta, x, n):

    price = []
    for k in range(n):
        x = np.exp(np.log(x)+norm.rvs(scale = delta))
        price.append(x)
    plt.plot(price)
    plt.xlabel('days')
    plt.ylabel("EUR")


# In[193]:


plot_lognorm_price(0.015, 13, 10000)


# In[ ]:




