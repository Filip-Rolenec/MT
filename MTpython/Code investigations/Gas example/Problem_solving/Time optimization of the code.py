#!/usr/bin/env python
# coding: utf-8

# # Trying to optimize my code 
# - The integral size sample computing the expected value is super slow, i need to make it faster 

# In[1]:


import sys
import os

sys.path.append("/Users/filiprolenec/Desktop/MT/MTpython/src")


# In[2]:


import operator
from typing import List

import numpy as np

from gas_example.enum_types import Action, PowerplantState
from gas_example.optimization.basis_function import uf_2, uf_2_inv
from gas_example.setup import BORROW_RATE_EPOCH, RISK_FREE_RATE_EPOCH
from gas_example.simulation.state import get_valid_actions, action_is_invalid, get_installed_mw, get_next_plant_state

from gas_example.setup import get_epoch_rate, GAS_VOL, CO2_VOL, POWER_VOL,     POWERPLANT_COST, MAINTENANCE_COST_PER_MW, HOURS_IN_EPOCH, BORROW_RATE_YEAR, RISK_FREE_RATE_YEAR, BORROW_RATE_EPOCH,     RISK_FREE_RATE_EPOCH, EPOCHS_IN_YEAR

import seaborn as sns
from os import listdir
from os.path import isfile, join
import gas_example.simulation.strategy as strategy
sns.set()

from progressbar import progressbar


# In[7]:


class State:
    def __init__(self, gas_price: float, co2_price: float, power_price: float,
                 plant_state: PowerplantState, balance: float):
        self.gas_price = gas_price
        self.co2_price = co2_price
        self.power_price = power_price
        self.plant_state = plant_state
        self.balance = balance
        
        self.price_coefs_dict = get_price_coefs_dict(INTEGRAL_SAMPLE_SIZE)

    # Used to get faster expected utility, vector form giving us faster results
    def get_spark_prices_and_fcfs(self, action):
        gas_prices = self.gas_price*self.price_coefs_dict["Gas"]
        co2_prices = self.co2_price*self.price_coefs_dict["CO2"]
        power_prices = self.power_price*self.price_coefs_dict["Power"]
    
        
        plant_state = [get_next_plant_state(self, action)]*INTEGRAL_SAMPLE_SIZE

        spark_prices = power_prices-co2_prices-gas_prices
        
        fcfs = compute_fcfs(spark_prices,
                          self.plant_state,
                          action)

        return spark_prices, fcfs
    
    # this is for the actual simulation.
    def get_new_state_and_fcf(self, action):
        gas_price = get_next_price(self.gas_price, GAS_VOL)
        co2_price = get_next_price(self.co2_price, CO2_VOL)
        power_price = get_next_price(self.power_price, POWER_VOL)
        plant_state = get_next_plant_state(self, action)

        fcf = compute_fcf(self,
                          self.plant_state,
                          action)

        balance = round(update_balance(self.balance, fcf))

        return State(gas_price, co2_price, power_price, plant_state, balance), fcf
    
    def to_dict(self):
        return self.__dict__
    
    def get_spark_price(self):
        return self.power_price - self.co2_price - self.gas_price - MAINTENANCE_COST_PER_MW


# In[8]:


def get_best_action(state: State, future_vf, print_details=False):
    valid_actions = get_valid_actions(state)

    exp_utility_per_action = {}
    for action in valid_actions:
        # We would like to compute expected value, we approximate by average of samples.
        utility_realizations = get_utility_realizations(state, action, future_vf)

        exp_utility_per_action[action] = np.mean(utility_realizations)
        
    if print_details:
        print(f"Spark: {state.get_spark_price()}")
        print(exp_utility_per_action)
        print("\n")

    best_action = max(exp_utility_per_action.items(), key=operator.itemgetter(1))[0]

    return best_action, exp_utility_per_action[best_action]


# In[ ]:





# In[9]:


[f for f in listdir("../saved_vfs") if isfile(join("../saved_vfs", f))]


# In[10]:


opt_strategy = strategy.OptimalStrategy("../saved_vfs/vfs_2020-12-28_H18.pkl")


# In[11]:


CO2_VOL = 0.10
GAS_VOL = 0.12
POWER_VOL = 0.15
EPOCHS_IN_YEAR = 12


# In[12]:


def get_price_coefs_dict(number_of_samples):
    asset_names = ["CO2", "Gas", "Power"]
    sigmas = [CO2_VOL, GAS_VOL, POWER_VOL]
    price_coefs_dict = {}

    for i, name in enumerate(asset_names):
        dt = 1/EPOCHS_IN_YEAR
        price_coefs_dict[name] = np.exp((0 - sigmas[i] ** 2 / 2) * dt + sigmas[i] * np.random.normal(0, np.sqrt(dt), number_of_samples))

    return price_coefs_dict


# In[14]:


def compute_fcfs(spark_prices, 
                plant_state: PowerplantState,
                action: Action):
    
    single_profit = 0

    # Building new capacity costs money
    if action == Action.IDLE_AND_BUILD or action == Action.RUN_AND_BUILD:
        single_profit = - POWERPLANT_COST

    installed_mw = get_installed_mw(plant_state)


    single_profit -= installed_mw * MAINTENANCE_COST_PER_MW * HOURS_IN_EPOCH

    # Making profit if action is to run:
    profits = [single_profit]*INTEGRAL_SAMPLE_SIZE
    
    if action == Action.RUN_AND_BUILD or action == Action.RUN:
        profits += spark_prices * installed_mw * HOURS_IN_EPOCH

    return profits


# In[ ]:





# In[ ]:





# In[15]:


def get_utility_realizations(state: State, action: Action, future_vf):
    spark_prices, fcfs = state.get_spark_prices_and_fcfs(action)
    new_powerplant_state = get_next_plant_state(state, action)
    
    future_vf_utilities = future_vf.compute_value(new_powerplant_state, spark_prices)
    
    future_vf_money_equivalents = INVERSE_UTILITY_FUNCTION_V(future_vf_utilities)

    updated_balances = [fcf+state.balance for fcf in fcfs]
    
    balance_future_vf_pairs = [[a, b] for a,b in zip(updated_balances, future_vf_money_equivalents)]

    pce_realizations = [pce - state.balance for pce in pce_v(balance_future_vf_pairs)]

    utility_realizations = np.round(UTILITY_FUNCTION_V(pce_realizations), 2)
    
    return utility_realizations


# In[16]:


UTILITY_FUNCTION_V = np.vectorize(uf_2)
INVERSE_UTILITY_FUNCTION_V = np.vectorize(uf_2_inv)


# In[17]:


def pce(fcfs):
    balance = 0

    r_b = BORROW_RATE_EPOCH
    r_r = RISK_FREE_RATE_EPOCH

    for fcf in fcfs:
        balance += fcf
        if balance < 0:
            balance = balance * r_b
        else:
            balance = balance * r_r

    if balance < 0:
        return balance / r_b ** (len(fcfs))
    else:
        return balance / r_r ** (len(fcfs))


# In[18]:


def pce_v(fcfs_v): 
    pces = []
    for fcfs in fcfs_v: 
        pces.append(pce(fcfs))
    return pces


# In[20]:


vfs = opt_strategy.vfs


# In[27]:


INTEGRAL_SAMPLE_SIZE = 100


# In[28]:


init_state=State(10, 25, 400, PowerplantState.STAGE_1, 0)


# In[30]:


get_utility_realizations(init_state,Action.RUN, vfs[20])


# In[31]:


vfs = opt_strategy.vfs


# In[43]:


import pendulum
import pandas as pd


# In[44]:


pendulum.now()


# In[45]:


for i in range(10): 
    print(i)
    before_time = pd.Timestamp.now()
    INTEGRAL_SAMPLE_SIZE = 10**i
    init_state=State(10, 25, 400, PowerplantState.STAGE_1, 0)

    get_best_action(init_state, opt_strategy.vfs[1], True)
    print((pd.Timestamp.now()-before_time))


# In[48]:


0.19*200*300/60/60


# In[ ]:


INTEGRAL_SAMPLE_SIZE = 10000
get_best_action(init_state, opt_strategy.vfs[1], True)


# In[ ]:




