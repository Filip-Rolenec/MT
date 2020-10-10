#!/usr/bin/env python
# coding: utf-8

# # ADP algorihtm 
# - The goal of this file is to first learn how does the ADP work on a simple example and then use it to solve the problem of gas power plant valuation. 
# 
# - First, I will setup a simple example of three states and two actions. Each of the two actions changes the probability distributions of results and "costs" some reward. 
#     - I will compute the optimal strategy for this example with real dynamic programming and then with the approximative dynamic programming. 
#     - I will make heuristic strategies as well. 

# ## Three states two actions

# In[1]:


import sys
import os
sys.path.append("/Users/filiprolenec/Desktop/MT/MTpython/src")


# In[4]:


import simple_example.strategy as s
from simple_example.simulation import run_simulation
import matplotlib.pyplot as plt 
import numpy as np
from simple_example.setup import SimpleProblemSetup
from sklearn.linear_model import LinearRegression
from simple_example.state import get_new_state


# # Classical Dynamic programming 

# In[5]:


from simple_example.dp_algorithm import classic_dp


# In[7]:


horizon_vf = [0,0,0]
problem_setup = SimpleProblemSetup()
prob_matrix = problem_setup.prob_matrix
reward_matrix = problem_setup.reward_matrix
time_epochs = problem_setup.time_epochs


# In[8]:


classic_dp(horizon_vf, problem_setup)


# # Strategy result comparison

# In[9]:


strategies = [s.heuristic_strategy_0, 
              s.heuristic_strategy_1, 
              s.heuristic_strategy_2, 
              s.optimal_strategy]


# In[10]:


all_results = {}
initial_state = 0

for strategy in strategies: 
    strategy_results = {}
    for i in range(10000): 
        strategy_results[i] = run_simulation(strategy, initial_state, problem_setup)
            
    all_results[strategy.__name__] = strategy_results 


# In[11]:


for strategy in strategies: 
    plt.hist(all_results[strategy.__name__].values(), label =strategy.__name__, alpha = 0.7)
plt.legend()


# In[12]:


for strategy in strategies: 
    print(np.mean(list(all_results[strategy.__name__].values())))


# # ADP algorithm finding the best strategy 
# $$V_t(s, \theta) = \theta_0 + \theta_{t,1} \cdot \phi_1(s) + \theta_{t,2} \cdot \phi_2(s) + \theta_{t,3} \cdot \phi_3(s)$$

# Where $\phi_i(s)$ functions are onli indicator functions of each state.

# ## ADP steps: 
# 1. Initialize parameters $\theta_t$
# 2. Define basis functions $\phi_i(s)$
# 3. Loop over time epochs from the last to the first: 
#     1. Loop over fixed amount of sampled states: 
#         1. Determine optimal decisions based on those states, in my case 1 action out of 6. 
#         2. Simulate what happens after my action (s,a,s) triplet and reward (plus value function estimate in the following state) 
#     2. Make a linear regression on the new findings. Get new $\theta_i$ based on rewards + future vf and values of the basis functions. 
#     3. Update the parameters, either fully, or with a learning step. 
# 4. Check how far are the value functions from the truth (which I know here) 

# ### 1. initialize params

# In[13]:


thetas = [[0 for i in range(3)] for i in time_epochs]


# In[14]:


thetas[0]


# In[15]:


theta_initial = [0,0,0,0]


# ### 2. Define basis functions

# In[16]:


def bf_1(s): 
    return 1 if s == 0 else 0 

def bf_2(s): 
    return 1 if s == 1 else 0 

def bf_3(s): 
    return 1 if s == 2 else 0 
basis_functions = [bf_1,bf_2,bf_3]


# ### 3.a determine optimal decision based on a state and simulate what happens 

# In[17]:


def vector_mult(a,b): 
    return sum([i*j for i,j in zip(a,b)])


# In[18]:


state = 2
action = 1
actions = range(2)
states = range(3)
t = 9


# In[19]:


prob_matrix[2][1]


# In[20]:


problem_setup.reward_matrix[2][1]


# In[21]:


vector_mult(prob_matrix[2][1], reward_matrix[2][1])


# In[22]:


vf = []


# In[23]:


from simple_example.vf import Vf


# In[24]:


def create_vfs(time_epochs, theta_initial):
    vfs = []
    for i in time_epochs: 
        vfs.append(Vf(theta_initial))
    return


# In[28]:


def get_exp_value(state, action, prob_matrix, reward_matrix, future_vf): 
    rewards = reward_matrix[state][action]
    future_vf_values = future_vf.compute_all_values()
    total_rewards = [i+j for i,j in zip(rewards, future_vf_values)]
    return vector_mult(prob_matrix[state][action], total_rewards) 


# In[29]:


def get_best_action(state, future_vf, prob_matrix, reward_matrix): 
    exp_rewards = []
    for action in actions: 
        exp_rewards.append(get_exp_value(state, action, prob_matrix, reward_matrix, future_vf))
    return np.argmax(exp_rewards)


# ### 3.b for x random states, determine the best action, perform the evolution, get the actual reward and note it. 

# In[ ]:





# In[30]:


sample_size = 50


# In[31]:


def get_state_reward_pairs(sample_size, states, future_vf, prob_matrix, reward_matrix): 
    state_reward_pairs = []
    for i in range(sample_size): 
        state = np.random.choice(states, p=[0.34, 0.33, 0.33])

        action = get_best_action(state, future_vf, prob_matrix, reward_matrix) 
        new_state = get_new_state(state, action, prob_matrix)

        reward = reward_matrix[state][action][new_state] + future_vf.compute_value(new_state)
        state_reward_pairs.append([state, reward])
    return state_reward_pairs 


# ### 3.c linear regression on the results 
# - I have the following model: 
# $$V_t(s, \theta) = \theta_0 + \theta_{t,1} \cdot \phi_1(s) + \theta_{t,2} \cdot \phi_2(s) + \theta_{t,3} \cdot \phi_3(s)$$
# - and I want to predict the parameters \theta, based on the realizaitons V_t(s) and the variable s. 
# - This model is simple, and we expect that each of the parameters will be close to the expected value. 
#     - In reality, it will be such a number that minimizes the sum of squares of the errors. 
#     - For example if there are three outcomes with 1/3 probability and rewards 1,3,100, the number will be closer to 50 than 33. 
#     - Thus even for this model, the result will in uneven settings of reeward be only an approximation. 

# ### 3.c Intercept makes the model crazy 
# - There are numbers like 10^14 
# - Since there is no reason for the intercept, i will just make a model without it here. The model becomes: 
# 
# $$V_t(s, \theta) = \theta_{t,1} \cdot \phi_1(s) + \theta_{t,2} \cdot \phi_2(s) + \theta_{t,3} \cdot \phi_3(s)$$
# 

# preparing the regression variables

# In[32]:


def prepare_regression_variables(state_reward_pairs, basis_functions): 
    x = []
    y = []
    for pair in state_reward_pairs: 
        x.append([basis_functions[0](pair[0]),
                  basis_functions[1](pair[0]),
                  basis_functions[2](pair[0])])
        y.append(pair[1])
        
    return x,y


# actual values = 6.4, 13.8, 14.4

# ### But the actual prediction makes sense i guess 

# In[33]:


def update_vf_coef(current_vf, next_vf, problem_setup, sample_size, basis_functions):
    state_reward_pairs_raw = get_state_reward_pairs(sample_size, 
                                                    problem_setup.states, 
                                                    next_vf,
                                                    problem_setup.prob_matrix,
                                                    problem_setup.reward_matrix)
    
    x,y = prepare_regression_variables(state_reward_pairs_raw, basis_functions)
    model = LinearRegression(fit_intercept=False).fit(x, y)
    current_vf.set_params(model.coef_)


# 106.58909091199999, 113.321546176, 113.26535516799999

# Even with 5000 samples and 20 loops of update, there is still a 0.5% difference from the reality. This might be caused by a different optimization function as discussed above. Linear algorithm does not return the mean value, rather a value from which the sum of squares of residuals is the lowest. 
# 
# A reasonable approximation can be seen as low as for 50 samples and 10 loops. 

# In[37]:


problem_setup = SimpleProblemSetup()


# In[38]:


from simple_example.adp_algorithm import adp_algorithm_final


# In[39]:


vfs_1 = adp_algorithm_final(loops_of_update=20, sample_size=50, problem_setup = problem_setup)


# In[40]:


vfs_1[0].params


# 106.58909091199999, 113.321546176, 113.26535516799999

# In[41]:


vfs_1 = adp_algorithm_final(loops_of_update=100, sample_size=2000, problem_setup = problem_setup)


# In[42]:


vfs_1[0].params


# 106.58909091199999, 113.321546176, 113.26535516799999

# # Conclusion 
# - I have implemented the ADP algorithm on the simple example. 
# - We can see that the approximation is really good in this example. The implementation is working. 
# - This investigation will help with the implementation of the actual problem in the next phase. 
#     - The complexity will rise significantly, each of the steps will be somehow harder and the computational complexity will rise too. Nevertheless, the framework is ready to be used. 

# 
# # That was investigation for simple model 

# # Now lets discuss problems of ADP of the gas powerplant example
# 

# # How to create random sample states in time t? 
# - For price of gas I know the distribution, that distribution is just the lognormal process in time. 

# ## Sampling price of gas and co2 prices. 
# - Here the distribution is known. We know the behavior of the variables and log-normal process has a nice feature. 

# In[ ]:


x2 = np.exp(np.log(x1)+norm_1)
x3 = np.exp(np.log(x2)+norm_2)

x3 = np.exp(np.log(np.exp(np.log(x1)+norm_1))+norm_2)
x3 = np.exp(np.log(x1)+norm_1+norm_2)

x_t = np.exp(np.log(x1)+sum_{i=1, t-1} norm_i)


# Sum of normal variables has nice properties [here](https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables)
# - I have zero means, but I will need to multiply the volatility. Lets check it out. 

# In[136]:


from scipy.stats import norm


# In[137]:


def get_gas_price_in_t(sigma, x_start, time_epoch, sample_size):

    prices = []
    for i in range(sample_size): 
        x = x_start
        for k in range(time_epoch):
            x = np.exp(np.log(x)+np.random.normal(0, sigma, 1)[0])
        prices.append(x)
    return prices


# In[138]:


def get_gas_price_one_step(sigma, x_start, time_epochs, sample_size): 
    prices = [] 
    for i in range(sample_size): 
        x = np.exp(np.log(x_start)+ np.random.normal(0, sigma*np.sqrt(time_epochs), 1)[0])
        prices.append(x)
    return prices


# In[142]:


step_by_step = get_gas_price_in_t(0.015, 40, 100, 10000)


# In[143]:


one_step_results = get_gas_price_one_step(0.015, 40,100,10000)


# In[145]:


plt.hist(step_by_step, label ="1", alpha = 0.7, bins = 50)
plt.hist(one_step_results, label ="2", alpha = 0.7, bins = 50)
plt.show()


# This is ok 

# ## Sampling of prices of power. 
# - Here it gets a little bit more complicated. 
# - The volatility of power price depends on the government politics. 
# - What we can do is to determine the average expected volatility through that time and compare. 

# In[163]:


prob_down = 0.04 
prob_up = 0.08 
zero_prob = 1 - prob_down - prob_up


# In[167]:


def get_next_gov_state(gov_state, prob_up, prob_down):
    
    zero_prob = 1 - prob_up - prob_down
    current_move = np.random.choice(np.arange(1, 4), p=[prob_down, zero_prob, prob_up])-2

    if gov_state+current_move in range(1,6): 
        gov_state += current_move
    return gov_state
    


# In[170]:


def get_power_price_in_t(sigma_original, x_start, time_epoch, sample_size, prob_down, prob_up):

    prices = []
    for i in range(sample_size): 
        x = x_start
        gov_policy = 1
        for k in range(time_epoch):
            sigma = sigma_original*(1+(gov_policy-1)*0.2)
            x = np.exp(np.log(x)+np.random.normal(0, sigma, 1)[0])
            gov_policy = get_next_gov_state(gov_policy, prob_up, prob_down)
        prices.append(x)
    return prices


# In[229]:


step_by_step = get_power_price_in_t (0.01, 40, 100, 10000, prob_down, prob_up)


# The following computation does not hold for such time epochs, where the value is 5. The value is never five. 

# In[230]:


def get_exp_gov_policy_in_t(prob_up, prob_down, epoch): 

    return min(1+epoch*(prob_up-prob_down), 5) 


# In[231]:


def get_avg_exp_policies(prob_up, prob_down, epochs): 
    return[get_exp_gov_policy_in_t(prob_up,prob_down,i) for i in range(epochs)]


# In[247]:


exp_policies = get_avg_exp_policies(prob_up, prob_down, 100)
sigmas = [0.01*(1+(gov_policy-1)*0.2) for gov_policy in exp_policies]
sigmas_squared = [sigma*sigma for sigma in sigmas]
sigma_average = np.sqrt(sum(sigmas_squared))/10


# This looks like a reasonable average sigma with which I could compute the power distribution

# In[248]:


one_step_result = get_gas_price_one_step(sigma_average, 40, 100, 10000)


# In[249]:


plt.hist(step_by_step, label ="1", alpha = 0.7, bins = 50)
plt.hist(one_step_result, label ="2", alpha = 0.7, bins = 50)
plt.legend()
plt.show()


# Looks also very reasonable 

# ## Government policy 
# - Could also be computed, probably has something like a bounded binomial distribution or something like that. 
# - What I will do is to get a reasonable amount of samples for all cases of government policy, with the majority around the expected value. 

# In[ ]:


epoch = 50


# In[275]:


def get_gov_samples(epoch, prob_up, prob_down, sample_size):
    exp_gov_policy = 1+(prob_up-prob_down)*epoch
    raw_random_policies = np.random.normal(exp_gov_policy, 1, sample_size)    
    return clean_policy_distribution(raw_random_policies)


# In[276]:


def clean_policy_distribution(raw_random_policies): 
    clean_policies = []
    for policy in raw_random_policies: 
        if policy<1: 
            clean_policies.append(1)
        elif policy>5: 
            clean_policies.append(5)
        else: 
            clean_policies.append(round(policy))
    return clean_policies 


# In[278]:


clean_policies = get_gov_samples(35, prob_up, prob_down, sample_size)


# In[281]:


plt.hist(clean_policies, bins = 5)
plt.legend()
plt.show()


# # Running state and powerplant states 
# - Will be done heuristically 
#     - Not build 5%
#     - Stage 1 40%
#     - Stage 2 50 % 
#     - Sold 5%. 
#     
#     
#     - Running 50% 
#     - Not running 45% 
#     - Mothballed 5% 

# # Balance 
# - some small number of realizations from -130M to 130M 
# 

# # Now everything at once
# 

# In[349]:


epoch = 50 
sample_size = 30

prob_up = 0.08
prob_down = 0.04 

gas_price = 13
gas_volatility = 0.04

co2_price = 9 
co2_volatility = 0.025

power_price =40 
power_volatility = 0.06


# In[350]:


def get_lognormal_prices(start_price, time_epoch, sigma, sample_size): 
    return [np.exp(np.log(start_price)+ i) for i in np.random.normal(0, sigma*np.sqrt(time_epoch), sample_size)]


# In[352]:


def get_avg_sigma_for_price(prob_up, prob_down, epoch, sigma):
    exp_policies_by_epoch = get_avg_exp_policies(prob_up, prob_down, epoch)
    sigmas = [sigma*(1+(epoch_policy-1)*0.2) for epoch_policy in exp_policies_by_epoch]
    sigmas_squared = [sigma*sigma for sigma in sigmas]
    sigma_average = np.sqrt(sum(sigmas_squared)/epoch)
    return sigma_average


# In[358]:


def get_power_price_sample(prob_up, prob_down, epoch, power_volatility, sample_size, power_price): 
    avg_sigma = get_avg_sigma_for_price(prob_up, prob_down, epoch, power_volatility)
    return get_lognormal_prices(power_price, epoch, avg_sigma, sample_size)


# In[370]:


from gas_example.enum_types import PowerplantState
from gas_example.enum_types import RunningState


# In[367]:


def get_powerplant_state_sample(sample_size): 
    states = [] 
    for i in range(sample_size): 
        states.append(np.random.choice([PowerplantState.NOT_BUILT, 
                      PowerplantState.STAGE_1,
                      PowerplantState.STAGE_2, 
                      PowerplantState.SOLD], 
                     p=[0.05, 0.4, 0.5, 0.05 ]))
    return states


# In[371]:


def get_running_state_sample(sample_size): 
    states = [] 
    for i in range(sample_size): 
        states.append(np.random.choice([RunningState.RUNNING, 
                                        RunningState.NOT_RUNNING, 
                                        RunningState.MOTHBALLED], 
                                       p=[0.5, 0.45, 0.05]))
    return states


# Will be done heuristically
# Not build 5%
# Stage 1 40%
# Stage 2 50 %
# Sold 5%.
# - Running 50% 
# - Not running 45% 
# - Mothballed 5% 

# In[374]:


def get_balance_sample(sample_size): 
    return np.random.uniform(-130_000_000,130_000_000,sample_size)


# In[377]:


def get_individual_samples(): 
    gas_price_sample = get_lognormal_prices(gas_price, epoch, gas_volatility, sample_size)
    co2_price_sample = get_lognormal_prices(co2_price, epoch, co2_volatility, sample_size)
    power_sample = get_power_price_sample(prob_up, prob_down, epoch, power_volatility, sample_size, power_price)
    clean_policies = get_gov_samples(epoch, prob_up, prob_down, sample_size)
    powerplant_state = get_powerplant_state_sample(sample_size)
    running_states = get_running_state_sample(sample_size)
    balance_sample = get_balance_sample(sample_size)
    
    return samples = [gas_price_sample, 
          co2_price_sample, 
          power_sample, 
          clean_policies, 
          powerplant_state, 
          running_states, 
          balance_sample]


# # It is not possible to take a cartesian product of all of the individual samples. 
# - We need to take a subsample, for example a 1000 rounds of choosing randomly a sample from each category 

# In[379]:


samples = [gas_price_sample, 
          co2_price_sample, 
          power_sample, 
          clean_policies, 
          powerplant_state, 
          running_states, 
          balance_sample]


# In[ ]:





# In[383]:


sample_size_global = 1000


# In[388]:


def get_global_sample(sample_size_global, samples): 
    states = []
    for i in range(sample_size_global): 
        state = []
        for sample in samples: 
            state.append(sample[np.random.choice(range(sample_size))])
        states.append(state)
    return states


# In[391]:


get_global_sample(sample_size_global, samples)


# In[ ]:




