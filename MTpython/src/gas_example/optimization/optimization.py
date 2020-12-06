from typing import List

from gas_example.simulation.state import State


def get_state_reward_pairs(
        sampled_states: List[State],
        epoch: int,
        future_vf

):
    state_reward_pairs = []
    for state in sampled_states:
        action = get_best_action(state, 10, epoch, future_vf)
        new_state, fcf = state.get_new_state_and_fcf(action, epoch)

        # TODO add utility and PCE here
        # Lets say that the reward is obtained from s_t, but it arrives at the end of a month, so that the general formula fits.
        value_new_state = future_vf.compute_value(new_state)

        # Present cash equivalent, balance now, fcf+value_new_state 1 epoch later.
        pce_value_new_state = pce([state.balance, fcf + value_new_state])

        # utility of the PCE
        utility_of_new_state = utility_function(pce_value_new_state)

        state_reward_pairs.append([state, utility_of_new_state])
    return state_reward_pairs


def get_best_action(state: State, integral_sample_size: int, epoch: int, future_vf):
    valid_actions = get_valid_actions(state, epoch)

    exp_rew_per_action = {}
    for action in valid_actions:
        sample_integral_values = []
        for i in range(integral_sample_size):
            new_state, reward = state.get_new_state_and_reward(action, epoch)
            vf_value = future_vf.compute_value(new_state)
            sample_integral_values.append(reward + vf_value)

        exp_rew_per_action[action] = np.mean(sample_integral_values)

    return max(exp_rew_per_action.items(), key=operator.itemgetter(1))[0]


# General pce function, used when the entity that is being optimized is not expected cumulative FCF.
def pce(fcfs, r_r, r_b):
    balance = 0
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
