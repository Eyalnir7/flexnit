import time
from itertools import combinations
from poop import MiniPlanner

import numpy as np
import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
CNT_TOTAL = 0
AVG = 1

EXPLORED_ALL_ARMS = -1
NUM_ROUNDS = 10 ** 6
PHASE_LEN = 10 ** 2
TIME_CAP = 2 * (10 ** 2)


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
        """
        :input: the instance parameters (see explanation in MABSimulation constructor)
        """
        # TODO: Decide what/if to store. Could be used in the future
        self.num_rounds = num_rounds
        self.phase_len = phase_len
        self.num_arms = num_arms
        self.num_users = num_users
        self.arms_thresh = arms_thresh
        self.users_distribution = users_distribution
        self.current_round = 0
        self.users_dict = {}
        # for all user, we'll have the history of how much times
        # the arm was picked for him, and what is his average gain from it
        for i in range(num_users):
            self.users_dict[i] = np.zeros((self.num_arms, 2))
        self.arms_phases = np.zeros((self.num_arms, 2))  # same as users_dict but don't relate to users, and restart per phase
        self.last_arm = None
        self.last_user = None
        self.inactive = []
        self.end_exploration_round = np.floor((self.num_rounds/self.num_arms)**(2/3) * (np.log(self.num_rounds)**(1/3)))
        self.avg_iteration_speed = time.time()


    def UCB(self, arms_arr):
        """
        :param arms_arr: list of arms that are available
        :return: the arm with max ucb value from the available arms
        """
        max_UCB = -1
        max_arm = -1
        for arm in range(self.num_arms):
            if arm not in self.inactive and arm in arms_arr:
                UCB = self.users_dict[self.last_user][arm][AVG]
                UCB += np.sqrt(2 * np.log(self.num_rounds * self.users_distribution[self.last_user]) /
                               self.users_dict[self.last_user][arm][CNT_TOTAL])
                if UCB > max_UCB:
                    max_UCB = UCB
                    max_arm = arm
        return max_arm


    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        self.current_round += 1
        arm_to_return = None
        self.last_user = user_context

        unexplored_arm = self.next_arm_to_explore()
        if unexplored_arm != EXPLORED_ALL_ARMS:
            self.last_arm = unexplored_arm
            self.arms_phases[self.last_arm][CNT_TOTAL] += 1
            self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
            arm_to_return = self.last_arm
        else:
            arm_at_risk = self.check_risk_deactivation()
            if arm_at_risk != EXPLORED_ALL_ARMS:
                self.last_arm = arm_at_risk
                self.arms_phases[self.last_arm][CNT_TOTAL] += 1
                self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
                arm_to_return = self.last_arm
            else:
                max_arm = self.UCB(list(range(self.num_arms)))
                self.last_arm = max_arm
                self.arms_phases[self.last_arm][CNT_TOTAL] += 1
                self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
                arm_to_return = self.last_arm
        return arm_to_return

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # TODO: Use this information for your algorithm
        self.update_for_user(reward)
        self.update_for_all(reward)
        if self.current_round % self.phase_len == 0:
            self.deactivation()
        if self.current_round == self.end_exploration_round:
            self.avg_iteration_speed = (time.time() - self.avg_iteration_speed) / self.end_exploration_round
            best_subset = self.run_simulations()
            for arm in range(self.num_arms):
                if arm not in best_subset:
                    self.arms_thresh[arm] = 0


    def update_for_user(self, reward):
        if self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] == 1:
            self.users_dict[self.last_user][self.last_arm][AVG] = reward
            return

        last_avg = self.users_dict[self.last_user][self.last_arm][AVG]
        last_impressions = self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] - 1
        new_avg = (last_avg*last_impressions + reward) / (last_impressions+1)
        self.users_dict[self.last_user][self.last_arm][AVG] = new_avg

    def update_for_all(self, reward):
        if self.arms_phases[self.last_arm][CNT_TOTAL] == 1:
            self.arms_phases[self.last_arm][AVG] = reward
            return
        # Now I know that I have at least one record of last_arm, in the current phase
        last_temp_avg = self.arms_phases[self.last_arm][AVG]
        last_temp_impressions = self.arms_phases[self.last_arm][CNT_TOTAL] - 1
        new_temp_avg = (last_temp_avg*last_temp_impressions + reward) / (last_temp_impressions+1)
        self.arms_phases[self.last_arm][AVG] = new_temp_avg


    def next_arm_to_explore(self):
        for arm in range(self.num_arms):
            if arm not in self.inactive:
                if self.users_dict[self.last_user][arm][CNT_TOTAL] == 0:
                    return arm
        return EXPLORED_ALL_ARMS

    def check_risk_deactivation(self):
        arms_to_activate = []
        demands_array = [max(self.arms_thresh[arm]-self.arms_phases[arm][CNT_TOTAL], 0) for arm in range(self.num_arms)
                         if arm not in self.inactive]
        sum_thresholds = np.sum(demands_array)
        if self.current_round % self.phase_len >= self.phase_len - sum_thresholds:
            for arm in range(self.num_arms):
                if arm not in self.inactive and self.arms_phases[arm][CNT_TOTAL] < self.arms_thresh[arm]:
                    arms_to_activate.append(arm)
        if not arms_to_activate:
            return EXPLORED_ALL_ARMS

        return self.UCB(arms_to_activate)

    def deactivation(self):
        """
        Those who need to get deactivated, becomes inactive. Initiallizes the count of the phase
        """
        for arm in range(self.num_arms):
            if arm not in self.inactive:
                if self.arms_phases[arm][CNT_TOTAL] < self.arms_thresh[arm]:
                    self.inactive.append(arm)
        self.arms_phases = np.zeros((self.num_arms, 2))


    def get_simulation_results(self, params):
        sim = myMABSimulation(params['num_rounds'], params['phase_len'], params['num_arms'], params['num_users'],
                      params['arms_thresh'], params['users_distribution'], params['ERM'])
        planner = MiniPlanner(params['num_rounds'], params['phase_len'], params['num_arms'], params['num_users'],
                      params['arms_thresh'], params['users_distribution'])
        return sim.simulation(planner)


    def run_simulations(self):
        """
        runs simulations of 'self.end_exploration_round' number of rounds with MiniPlanner on a every subset of the arms
        :return: the subset of the arms that did the best in the simulation
        """
        simulations_params = self.get_simulations_params()
        maxValue = 0
        maxSubset = simulations_params[0]['arms']
        for sim_params in simulations_params:
            res = self.get_simulation_results(sim_params)
            if res > maxValue:
                maxSubset = sim_params['arms']
                maxValue = res
        return maxSubset


    def get_simulation_length(self):
        """
        :return: the optimal amount of rounds each simulation should have
        """
        numerator = TIME_CAP - self.avg_iteration_speed * self.num_rounds
        denominator = 2**self.num_arms * self.avg_iteration_speed
        return numerator/denominator


    def get_simulations_params(self):
        simulations_array = []

        def get_combinations(n):
            combinations_list = []
            numbers = list(range(0, n))
            for subset_size in range(1, n + 1):
                combinations_list.extend([list(comb) for comb in combinations(numbers, subset_size)])
            return combinations_list

        combinations_list = get_combinations(self.num_arms)
        for comb in range(len(combinations_list)):
            thresholds = []
            for arm in range(self.num_arms):
                if arm in combinations_list[comb]:
                    thresholds.append(self.arms_thresh[arm])

            expectations = []
            for user in range(self.num_users):
                user_expectations = []
                for arm in range(self.num_arms):
                    if arm in combinations_list[comb]:
                        user_expectations.append(self.users_dict[user][arm][AVG])
                expectations.append(user_expectations)

            simulations_array.append({
                'num_rounds': int(self.get_simulation_length()),
                'phase_len': self.phase_len,
                'num_arms': len(combinations_list[comb]),
                'num_users': self.num_users,
                'users_distribution': self.users_distribution,
                'arms_thresh': np.array(thresholds),
                'ERM': np.array(expectations),
                'arms': combinations_list[comb]}
            )

        return simulations_array

    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_213950496_325161669"


class myMABSimulation:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution, ERM):
        """
        :input: num_rounds - number of rounds
                phase_len - number of rounds at each phase
                num_arms - number of content providers
                num_users - number of users
                arms_thresh - the exposure demands of the content providers (np array of size num_arms)
                users_distribution - the probabilities of the users to arrive (np array of size num_users)
                ERM - expected reward matrix (2D np array of size (num_arms x num_users))
                      ERM[i][j] is the expected reward of user i from content provider j
        """
        self.num_rounds = num_rounds
        self.phase_len = phase_len
        self.num_arms = num_arms
        self.num_users = num_users
        self.arms_thresh = arms_thresh
        self.users_distribution = users_distribution
        self.ERM = ERM
        self.exposure_list = np.zeros(self.num_arms)
        # exposure_list[i] represents the number of exposures arm i has gotten in the current phase
        self.inactive_arms = set()  # set of arms that left the system

    def sample_user(self):
        """
        :output: the sampled user, an integer in the range [0,self.num_users-1]
        """
        return int(np.random.choice(range(self.num_users), size=1, p=self.users_distribution))

    def sample_reward(self, sampled_user, chosen_arm):
        """
        :input: sampled_user - the sampled user
                chosen_arm - the content provider that was recommended to the user
        :output: the sampled reward
        """
        if chosen_arm >= self.num_arms or chosen_arm in self.inactive_arms:
            return 0
        else:
            #print("User: " + str(sampled_user) + "Arm:" + str(chosen_arm) + "poopkaka" + str(self.ERM[sampled_user][chosen_arm]))
            return np.random.uniform(0, 2 * self.ERM[sampled_user][chosen_arm])

    def deactivate_arms(self):
        """
        this function is called at the end of each phase and deactivates arms that havn't gotten enough exposure
        (deactivated arm == arm that has departed)
        """
        for arm in range(self.num_arms):
            if self.exposure_list[arm] < self.arms_thresh[arm]:
                if arm not in self.inactive_arms:
                    self.inactive_arms.add(arm)
        self.exposure_list = np.zeros(self.num_arms)  # initiate the exposure list for the next phase.

    def simulation(self, planner, with_deactivation=True):
        """
        :input: the recommendation algorithm class implementation
        :output: the total reward for the algorithm
        """
        total_reward = 0
        begin_time = time.time()
        for i in range(self.num_rounds):
            user_context = self.sample_user()
            chosen_arm = planner.choose_arm(user_context)
            reward = self.sample_reward(user_context, chosen_arm)
            planner.notify_outcome(reward)
            total_reward += reward
            self.exposure_list[chosen_arm] += 1

            if (i + 1) % self.phase_len == 0 and with_deactivation:  # satisfied only when it is the end of the phase
                self.deactivate_arms()

        if time.time() - begin_time > TIME_CAP:
            print("the planner operation is too slow in semi simulation")
            return 0

        return total_reward

