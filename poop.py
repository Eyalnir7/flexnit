import numpy as np
import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
CNT_TOTAL = 0
AVG = 1

EXPLORED_ALL_ARMS = -1


class MiniPlanner:
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
        self.end_exploration_round = np.floor((self.num_rounds/self.num_arms)**(2/3) * (np.log(self.num_rounds)**(1/3))
                                              * self.num_users)


    def UCB(self, arms_arr):
        """
        :param arms_arr: list of arms that are available
        :return: the arm with max ucb value from the available arms
        """
        max_UCB = -1
        max_arm = -1
        for arm in range(self.num_arms):
            if arm not in self.inactive and arm in arms_arr:
                UCB = 0
                if self.users_dict[self.last_user][arm][CNT_TOTAL] > 0:
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

        arm_at_risk = self.check_risk_deactivation()
        if arm_at_risk != EXPLORED_ALL_ARMS:
            self.last_arm = arm_at_risk
            self.arms_phases[self.last_arm][CNT_TOTAL] += 1
            self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
            arm_to_return = self.last_arm
        else:
            unexplored_arm = self.next_arm_to_explore()
            if unexplored_arm != EXPLORED_ALL_ARMS:
                self.last_arm = unexplored_arm
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
