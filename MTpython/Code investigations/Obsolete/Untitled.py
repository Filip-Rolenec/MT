#!/usr/bin/env python
# coding: utf-8

# In[147]:


def uf_2(x):
    x = x / 1000

    if x <= 0:
        return -((-x) ** (1.3))
    else:
        return x ** (0.85)


# In[148]:


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# In[149]:


a  = np.linspace(-500000,500000,100001)


# In[150]:


b= [uf_2(x) for x in a]


# In[151]:


df = pd.DataFrame({"EUR":a, "Utility":b})


# In[152]:


fig = go.Figure()
fig.add_trace(go.Scatter(x=df["EUR"], y=df["Utility"],
                    mode='lines'))

fig.update_layout(xaxis_title="EUR", yaxis_title="Utility")


# In[ ]:





# In[ ]:




