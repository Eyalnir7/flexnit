import numpy as np
import pandas as pd

# import networkx as nx
# import matplotlib.pyplot as plt
CNT_TOTAL = 0
AVG = 1

EXPLORED_ALL_ARMS = -1
NEGATIVE_INFINITY = -10000000


class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution, ERM):
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
        self.erm = ERM
        self.current_round = 0
        self.phase_round = 0
        self.current_phase = 0
        self.users_dict = {}
        self.H = []
        # for all user, we'll have the history of how much times
        # the arm was picked for him, and what is his average gain from it
        for i in range(num_users):
            self.users_dict[i] = np.zeros((self.num_arms, 2))
        self.arms_phases = np.zeros(self.num_arms)

        self.last_arm = None
        self.last_user = None
        self.inactive = []
        self.bestZ, self.maxMER = self.get_Z()

    def get_Z(self):
        print("in getZ")
        subsets = [[]]
        for el in range(self.num_arms):
            subsets += [s + [el] for s in subsets]
        bestZ = subsets[0]
        print("trying first time mer")
        maxMER = self.MER([], bestZ)
        print("finished 1 mer")
        print(subsets)
        for Z in subsets:
            print("trying new sub set")
            curMER = self.MER([], Z)
            if curMER > maxMER:
                maxMER = curMER
                bestZ = Z
        return bestZ, maxMER

    def choose_arm(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        # TODO: Check the deactivation, each has its own factor of phase_len !!!
        if len(self.H) % self.phase_len == 0:
            self.H = []
        maxarm = 0
        maxval = NEGATIVE_INFINITY
        Ht = self.H.copy()
        for arm in range(self.num_arms):
            Ht.append(arm)
            curval = self.erm[user_context][arm] + self.MER(Ht, self.bestZ)
            if curval > maxval:
                maxval = curval
                maxarm = arm
        self.H.append(maxarm)
        return maxarm


    def MER(self, H, Z):
        """
        :param H: history of actions. represented as a list
        :param Z: subset of arms
        :return: The maximum expected reward
        TODO: optimize the array representation because I think it can greatly change the performance
        """
        if len(H) == self.phase_len:
            npH = np.array(H)
            for a in range(self.num_arms):
                if np.sum(npH[npH == a]) < self.arms_thresh[a] * self.phase_len:
                    return NEGATIVE_INFINITY
            return 0
        max_profit = NEGATIVE_INFINITY
        result = 0
        Hnew = H.copy()
        for u in range(self.num_users):
            for arm in range(self.num_arms):
                Hnew.append(arm)
                current = self.erm[u][arm] + self.MER(Hnew, Z)
                if current > max_profit:
                    max_profit = current
                Hnew.pop()
            result += self.users_distribution[u] * max_profit
        return result

    def notify_outcome(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # TODO: Use this information for your algorithm
        pass

    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_213950496_325161669"
