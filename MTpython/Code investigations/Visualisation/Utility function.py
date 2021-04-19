#!/usr/bin/env python
# coding: utf-8

# ## Example of a utility function 

# In[1]:


import plotly.graph_objects as go
import numpy as np

x = np.arange(201)

y = np.log(pow(x+20,0.2))*50-30

fig = go.Figure(data=go.Scatter(x=x, y=y))
fig.update_layout(xaxis_title = 'EUR',
                   yaxis_title='Utility')
fig.show()


# In[2]:


import plotly.io as pio
pio.write_image(fig, 'filename.pdf', width=700, height=775)


# In[ ]:




