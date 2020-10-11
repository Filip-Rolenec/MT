#!/usr/bin/env python
# coding: utf-8

# # ADP algorithm gas 
# - In this jupyter notebook I will be testing the functions for the final ADP algorithm for my gas powerplant example. 
# - When this file is completed, I will have a function which will be similar to the end of the investigation denoted as adp_algorithm_final. 

# In[1]:


import sys
import os
sys.path.append("/Users/filiprolenec/Desktop/MT/MTpython/src")


# In[2]:


from gas_example.sampling import get_state_sample
import numpy as np


# In[3]:


sample_size_global  = 10
sample_size_individual = 20
epoch = 35


# In[4]:


samples = get_state_sample(sample_size_global, sample_size_individual, epoch)


# In[5]:


def get_state_reward_pairs(future_vf, problem_setup, sampled_states):
    state_reward_pairs = []
    for state in sampled_states:
        
        action = get_best_action(state, future_vf, problem_setup)
        new_state = get_new_state(state, action, problem_setup.prob_matrix)

        reward = problem_setup.reward_matrix[state][action][new_state] + future_vf.compute_value(new_state)
        state_reward_pairs.append([state, reward])
    return state_reward_pairs


# In[6]:


sampled_states = samples[1]


# ### What action should I take? Based on the future value function and its model.
# - Get expected value for all actions.. 
# 

# ### Expected value for each action 
# - get_exp_value(state, action, problem_setup, future_vf)
# - rewards = problem_setup.reward_matrix[state][action]
# - future_vf_values = future_vf.compute_all_values()
# - total_rewards = [i + j for i, j in zip(rewards, future_vf_values)]
# - return vector_mult(problem_setup.prob_matrix[state][action], total_rewards)

# ### First idea 
# - Compute the expectation from an integral... 
# - 100 realizations of a future state, get a value out of each of them -> average is the expected value? 
# - since the sampling already respects the probabilities, there is no problem with the weighting. 

# ### Or 
# - I would need to compute integrals of the expectation. p(future_state) * functional expression of the future state. 
#     - p(future_state) could be okay, just a multiplication of some distributions. 
#     - reward as a functional expression - again could be done, but would not be pretty and in the end the integral would be computed numerically inside the program ... 

# # Lets go with the first idea and make a numerical approximation

# # Compute Reward 
# - As investigated before, the moneey that serve to repay the debt are discounted by the borrowing rate and the money that are earned are discounted by the risk_free rate. Now I need to be able to transform fcf and information about balance to the reward which is a cash equivalent in the first time epoch. 

# In[7]:


from gas_example.setup import GasProblemSetup


# In[8]:


balance = 40 
fcf = 100 
epoch = 30
risk_free_rate = 0.02
borrow_rate = 0.07 

epoch_rf_rate = risk_free_rate**(1/float(12))
epoch_b_rate = borrow_rate**(1/float(12))


# # Getting expected reward 
# - I need to make samples of future states, obtain the rewards in them and also compute Vf in them. 

# In[9]:


from gas_example.setup import GasProblemSetup
from gas_example.model import AdpModel
from gas_example.adp_algorithm import adp_algorithm_final


# In[10]:


ps = GasProblemSetup()
model = AdpModel()


# In[11]:


vfs = adp_algorithm_final(5, ps, model)


# In[12]:


from gas_example.state import get_initial_state


# In[13]:


vfs[0].compute_value(get_initial_state())


# # TODO 
# - **Check the complete algorithm**, lets print all steps. Find out the steps where the approximation is really bad. 
# - Run the algorithm precisely for many hours over the night. 
# - Play with the possible addition and removal of actions and different setups. 
# - Run monte carlo simulation, where decisions are made based on the trategy given by the value functions. 
# - Make the more reasonable model, where there is sold identificator as a multiplier. 
# - Check how many times mothballing is optimal action and if it makes sense. 
# - Run with more time epochs. 
# - **Make time optimization**, now I make 5 rounds of 300 time epochs, where I take 50 state samples and do a 10 samles for each in a numerical integration.
#     - 750k multiplications. 
#     - Linear model only 1500 times.  

# In[14]:


POWERPLANT_COST = 650_000_000


# In[15]:


from gas_example.fcf import compute_fcf
from gas_example.enum_types import RunningState, PowerplantState, Action


# In[16]:


vfs[0].params

