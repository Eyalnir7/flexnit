import numpy as np
import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
CNT_TOTAL = 0
AVG = 1

EXPLORED_ALL_ARMS = -1

class Planner1:
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
        self.arms_phases = np.zeros(self.num_arms)

        self.last_arm = None
        self.last_user = None
        self.inactive = []

    def choose_arm1(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        # TODO: Check the deactivation, each has its own factor of phase_len !!!
        self.current_round += 1
        arm_to_return = None

        self.last_user = user_context


        unexplored_arm = self.next_arm_to_explore1()
        if unexplored_arm != EXPLORED_ALL_ARMS:
            self.last_arm = unexplored_arm
            self.arms_phases[self.last_arm] += 1
            self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
            arm_to_return = self.last_arm
        else:
            arm_at_risk = self.check_risk_deactivation1()
            if arm_at_risk != EXPLORED_ALL_ARMS:
                self.last_arm = arm_at_risk
                self.arms_phases[self.last_arm] += 1
                self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
                arm_to_return = self.last_arm
            else:
                max_UCB = -1
                max_arm = -1
                for arm in range(self.num_arms):
                    if arm in self.inactive:
                        continue
                    UCB = self.users_dict[self.last_user][arm][AVG]
                    UCB += np.sqrt(2*np.log(self.num_rounds*self.users_distribution[self.last_user])/self.users_dict[self.last_user][arm][CNT_TOTAL])
                    if UCB > max_UCB:
                        max_UCB = UCB
                        max_arm = arm
                self.last_arm = max_arm
                self.arms_phases[self.last_arm] += 1
                self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
                arm_to_return = self.last_arm
        return arm_to_return

    def notify_outcome1(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # TODO: Use this information for your algorithm
        if self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] == 1:
            self.users_dict[self.last_user][self.last_arm][AVG] = reward
            return
        last_avg = self.users_dict[self.last_user][self.last_arm][AVG]
        last_choosing_amount = self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] - 1
        new_avg = (last_avg*last_choosing_amount + reward) / (last_choosing_amount+1)
        self.users_dict[self.last_user][self.last_arm][AVG] = new_avg
        if self.current_round % self.phase_len == 0:
            self.deactivation1()

    def next_arm_to_explore1(self):
        for arm in range(self.num_arms):
            if arm not in self.inactive and self.users_dict[self.last_user][arm][CNT_TOTAL] == 0:
                return arm
        return EXPLORED_ALL_ARMS

    def check_risk_deactivation1(self):
        arms_to_activate = []
        sum_thresholds = np.sum(self.arms_thresh)
        if self.current_round % self.phase_len >= self.phase_len - sum_thresholds:
            for arm in range(self.num_arms):
                if arm not in self.inactive and self.arms_phases[arm] < self.arms_thresh[arm]:
                    arms_to_activate.append(arm)
        if not arms_to_activate:
            return EXPLORED_ALL_ARMS

        max_UCB = -1
        max_arm = -1
        for arm in arms_to_activate:
            if arm in self.inactive:
                continue
            UCB = self.users_dict[self.last_user][arm][AVG]
            UCB += np.sqrt(2 * np.log(self.num_rounds*self.users_distribution[self.last_user]) / self.users_dict[self.last_user][arm][CNT_TOTAL])
            if UCB > max_UCB:
                max_UCB = UCB
                max_arm = arm
        return max_arm

    def deactivation1(self):
        """
        Those who need to get deactivated, becomes inactive. Initiallizes the count of the phase
        """
        for arm in range(self.num_arms):
            if arm not in self.inactive:
                if self.arms_phases[arm] < self.arms_thresh[arm]:
                    self.inactive.append(arm)
                    print("The inactive: " + str(self.inactive) + ", in round" + str(self.current_round))
        self.arms_phases = np.zeros(self.num_arms)



    def get_id1(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_213950496_325161669"

class Planner2:
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
        self.arms_phases = np.zeros((self.num_arms, 2))
        self.new_arms_thresh = arms_thresh
        self.last_arm = None
        self.last_user = None
        self.inactive = []

    def choose_arm2(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        # TODO: Check the deactivation, each has its own factor of phase_len !!!
        self.current_round += 1
        arm_to_return = None
        self.last_user = user_context

        unexplored_arm = self.next_arm_to_explore2()
        if unexplored_arm != EXPLORED_ALL_ARMS:
            self.last_arm = unexplored_arm
            self.arms_phases[self.last_arm][CNT_TOTAL] += 1
            self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
            arm_to_return = self.last_arm
        else:
            arm_at_risk = self.check_risk_deactivation2()
            if arm_at_risk != EXPLORED_ALL_ARMS:
                self.last_arm = arm_at_risk
                self.arms_phases[self.last_arm][CNT_TOTAL] += 1
                self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
                arm_to_return = self.last_arm
            else:
                max_UCB = -1
                max_arm = -1
                for arm in range(self.num_arms):
                    if arm in self.inactive:
                        continue
                    UCB = self.users_dict[self.last_user][arm][AVG]
                    UCB +=np.sqrt(2 * np.log(self.num_rounds) / self.users_dict[self.last_user][arm][CNT_TOTAL])
                    if UCB > max_UCB:
                        max_UCB = UCB
                        max_arm = arm
                self.last_arm = max_arm
                self.arms_phases[self.last_arm][CNT_TOTAL] += 1
                self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
                arm_to_return = self.last_arm
        return arm_to_return

    def notify_outcome2(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # TODO: Use this information for your algorithm
        self.update_for_user2(reward)
        self.update_for_all2(reward)
        if self.current_round % self.phase_len == 0:
            self.deactivation2()


    def update_for_user2(self, reward):
        if self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] == 1:
            self.users_dict[self.last_user][self.last_arm][AVG] = reward
            return

        last_avg = self.users_dict[self.last_user][self.last_arm][AVG]
        last_impressions = self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] - 1
        #print("last impressions of arm " + str(self.last_arm) + ": " + str(last_impressions) + " ROUND " + str(self.current_round))
        new_avg = (last_avg*last_impressions + reward) / (last_impressions+1)
        self.users_dict[self.last_user][self.last_arm][AVG] = new_avg

    def update_for_all2(self, reward):
        if self.arms_phases[self.last_arm][CNT_TOTAL] == 1:
            self.arms_phases[self.last_arm][AVG] = reward
            return
        # Now I know that I have at least one record of last_arm, in the current phase
        last_temp_avg = self.arms_phases[self.last_arm][AVG]
        last_temp_impressions = self.arms_phases[self.last_arm][CNT_TOTAL] - 1
        #print("last temp impressions of arm " + str(self.last_arm) + ": " + str(last_temp_impressions))
        #print(self.arms_phases[self.last_arm][CNT_TOTAL])
        new_temp_avg = (last_temp_avg*last_temp_impressions + reward) / (last_temp_impressions+1)
        self.arms_phases[self.last_arm][AVG] = new_temp_avg
        #print(self.arms_phases)


    def next_arm_to_explore2(self):
        for arm in range(self.num_arms):
            if arm not in self.inactive:
                if self.users_dict[self.last_user][arm][CNT_TOTAL] == 0 or self.arms_phases[arm][CNT_TOTAL] == 0:
                    return arm
        return EXPLORED_ALL_ARMS

    def check_risk_deactivation2(self):
        arms_to_activate = []
        sum_thresholds = np.sum(self.new_arms_thresh)
        if self.current_round % self.phase_len >= self.phase_len - sum_thresholds:
            for arm in range(self.num_arms):
                if arm not in self.inactive and self.arms_phases[arm][CNT_TOTAL] < self.new_arms_thresh[arm]:
                    arms_to_activate.append(arm)
        if not arms_to_activate:
            return EXPLORED_ALL_ARMS

        max_UCB = -1
        max_arm = -1
        for arm in arms_to_activate:
            if arm in self.inactive:
                continue
            UCB = self.users_dict[self.last_user][arm][AVG]
            UCB += np.sqrt(2 * np.log(self.num_rounds) / self.users_dict[self.last_user][arm][CNT_TOTAL])
            if UCB > max_UCB:
                max_UCB = UCB
                max_arm = arm
        return max_arm

    def deactivation2(self):
        """
        Those who need to get deactivated, becomes inactive. Initiallizes the count of the phase
        """
        for arm in range(self.num_arms):
            if arm not in self.inactive:
                if self.arms_phases[arm][CNT_TOTAL] < self.arms_thresh[arm]:
                    self.inactive.append(arm)
                    #print("The inactive: " + str(self.inactive) + ", in round" + str(self.current_round))
        if self.current_round >= (self.num_rounds/self.num_arms)**(2/3) * (np.log(self.num_rounds)**(1/3)):
            self.reassigning_thresholds2()
        self.arms_phases = np.zeros((self.num_arms, 2))
        #print("Round: " + str(self.current_round))

    def reassigning_thresholds2(self):
        # Checking at the end of a phase. We want to know the contribution of a current phase
        # to the thresholds, so we should KNOW HOW MUCH AN ARM EARNED IN AVG IN THE CURRENT PHASE
        utilities = []
        for arm in range(self.num_arms):
            if arm in self.inactive:
                utilities.append(0)
                continue
            # sum_expectation = 0
            # for user in range(self.num_users):
            #     sum_expectation += self.users_distribution[user]*self.users_dict[user][arm][AVG]
            utilities.append(self.arms_phases[arm][AVG])
            #print("Arm score " + str(self.arms_phases[arm][AVG]))
        #print("Utilities: " + str(utilities))
        utilities = list(utilities/np.sum(utilities)) # in [0,1]
        new_thresholds = []
        for arm in range(self.num_arms):
            if arm in self.inactive:
                new_thresholds.append(0)
                continue
            new_thresholds.append((2*self.new_arms_thresh[arm]/self.phase_len + utilities[arm])/3)
        self.new_arms_thresh = list((new_thresholds / np.sum(new_thresholds))*self.phase_len)



    def get_id2(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_213950496_325161669"

class Planner:
    def __init__(self, num_rounds, phase_len, num_arms, num_users, arms_thresh, users_distribution):
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
        self.arms_phases = None
        self.new_arms_thresh = None

        if self.variance() > 0.1:
            self.my_planner = 2
            self.arms_phases = np.zeros((self.num_arms, 2))
            self.new_arms_thresh = arms_thresh
        else:
            self.my_planner = 1
            self.arms_phases = np.zeros(self.num_arms)

        self.last_arm = None
        self.last_user = None
        self.inactive = []

    def variance(self):
        sum1 = 0
        for user in range(self.num_users):
            sum1 += (self.users_distribution[user] - (1 / self.num_users)) ** 2
        variance = sum1 / self.num_users
        max_variance = (1 - (1 / self.num_users)) ** 2
        return variance / max_variance

    def choose_arm(self, user_context):
        if self.my_planner==1:
            return self.choose_arm1(user_context)
        else:
            return self.choose_arm2(user_context)

    def notify_outcome(self, reward):
        if self.my_planner == 1:
            self.notify_outcome1(reward)
        else:
            self.notify_outcome2(reward)
    def choose_arm1(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        # TODO: Check the deactivation, each has its own factor of phase_len !!!
        self.current_round += 1
        arm_to_return = None

        self.last_user = user_context


        unexplored_arm = self.next_arm_to_explore1()
        if unexplored_arm != EXPLORED_ALL_ARMS:
            self.last_arm = unexplored_arm
            self.arms_phases[self.last_arm] += 1
            self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
            arm_to_return = self.last_arm
        else:
            arm_at_risk = self.check_risk_deactivation1()
            if arm_at_risk != EXPLORED_ALL_ARMS:
                self.last_arm = arm_at_risk
                self.arms_phases[self.last_arm] += 1
                self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
                arm_to_return = self.last_arm
            else:
                max_UCB = -1
                max_arm = -1
                for arm in range(self.num_arms):
                    if arm in self.inactive:
                        continue
                    UCB = self.users_dict[self.last_user][arm][AVG]
                    UCB += np.sqrt(2*np.log(self.num_rounds*self.users_distribution[self.last_user])/self.users_dict[self.last_user][arm][CNT_TOTAL])
                    if UCB > max_UCB:
                        max_UCB = UCB
                        max_arm = arm
                self.last_arm = max_arm
                self.arms_phases[self.last_arm] += 1
                self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
                arm_to_return = self.last_arm
        return arm_to_return

    def notify_outcome1(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # TODO: Use this information for your algorithm
        if self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] == 1:
            self.users_dict[self.last_user][self.last_arm][AVG] = reward
            return
        last_avg = self.users_dict[self.last_user][self.last_arm][AVG]
        last_choosing_amount = self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] - 1
        new_avg = (last_avg*last_choosing_amount + reward) / (last_choosing_amount+1)
        self.users_dict[self.last_user][self.last_arm][AVG] = new_avg
        if self.current_round % self.phase_len == 0:
            self.deactivation1()

    def next_arm_to_explore1(self):
        for arm in range(self.num_arms):
            if arm not in self.inactive and self.users_dict[self.last_user][arm][CNT_TOTAL] == 0:
                return arm
        return EXPLORED_ALL_ARMS

    def check_risk_deactivation1(self):
        arms_to_activate = []
        sum_thresholds = np.sum(self.arms_thresh)
        if self.current_round % self.phase_len >= self.phase_len - sum_thresholds:
            for arm in range(self.num_arms):
                if arm not in self.inactive and self.arms_phases[arm] < self.arms_thresh[arm]:
                    arms_to_activate.append(arm)
        if not arms_to_activate:
            return EXPLORED_ALL_ARMS

        max_UCB = -1
        max_arm = -1
        for arm in arms_to_activate:
            if arm in self.inactive:
                continue
            UCB = self.users_dict[self.last_user][arm][AVG]
            UCB += np.sqrt(2 * np.log(self.num_rounds*self.users_distribution[self.last_user]) / self.users_dict[self.last_user][arm][CNT_TOTAL])
            if UCB > max_UCB:
                max_UCB = UCB
                max_arm = arm
        return max_arm

    def deactivation1(self):
        """
        Those who need to get deactivated, becomes inactive. Initiallizes the count of the phase
        """
        for arm in range(self.num_arms):
            if arm not in self.inactive:
                if self.arms_phases[arm] < self.arms_thresh[arm]:
                    self.inactive.append(arm)
                    print("The inactive: " + str(self.inactive) + ", in round" + str(self.current_round))
        self.arms_phases = np.zeros(self.num_arms)



    def get_id(self):
        # TODO: Make sure this function returns your ID, which is the name of this file!
        return "id_213950496_325161669"

    def choose_arm2(self, user_context):
        """
        :input: the sampled user (integer in the range [0,num_users-1])
        :output: the chosen arm, content to show to the user (integer in the range [0,num_arms-1])
        """
        # TODO: Check the deactivation, each has its own factor of phase_len !!!
        self.current_round += 1
        arm_to_return = None
        self.last_user = user_context

        unexplored_arm = self.next_arm_to_explore2()
        if unexplored_arm != EXPLORED_ALL_ARMS:
            self.last_arm = unexplored_arm
            self.arms_phases[self.last_arm][CNT_TOTAL] += 1
            self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
            arm_to_return = self.last_arm
        else:
            arm_at_risk = self.check_risk_deactivation2()
            if arm_at_risk != EXPLORED_ALL_ARMS:
                self.last_arm = arm_at_risk
                self.arms_phases[self.last_arm][CNT_TOTAL] += 1
                self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
                arm_to_return = self.last_arm
            else:
                max_UCB = -1
                max_arm = -1
                for arm in range(self.num_arms):
                    if arm in self.inactive:
                        continue
                    UCB = self.users_dict[self.last_user][arm][AVG]
                    UCB +=np.sqrt(2 * np.log(self.num_rounds) / self.users_dict[self.last_user][arm][CNT_TOTAL])
                    if UCB > max_UCB:
                        max_UCB = UCB
                        max_arm = arm
                self.last_arm = max_arm
                self.arms_phases[self.last_arm][CNT_TOTAL] += 1
                self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] += 1
                arm_to_return = self.last_arm
        return arm_to_return

    def notify_outcome2(self, reward):
        """
        :input: the sampled reward of the current round.
        """
        # TODO: Use this information for your algorithm
        self.update_for_user2(reward)
        self.update_for_all2(reward)
        if self.current_round % self.phase_len == 0:
            self.deactivation2()


    def update_for_user2(self, reward):
        if self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] == 1:
            self.users_dict[self.last_user][self.last_arm][AVG] = reward
            return

        last_avg = self.users_dict[self.last_user][self.last_arm][AVG]
        last_impressions = self.users_dict[self.last_user][self.last_arm][CNT_TOTAL] - 1
        #print("last impressions of arm " + str(self.last_arm) + ": " + str(last_impressions) + " ROUND " + str(self.current_round))
        new_avg = (last_avg*last_impressions + reward) / (last_impressions+1)
        self.users_dict[self.last_user][self.last_arm][AVG] = new_avg

    def update_for_all2(self, reward):
        if self.arms_phases[self.last_arm][CNT_TOTAL] == 1:
            self.arms_phases[self.last_arm][AVG] = reward
            return
        # Now I know that I have at least one record of last_arm, in the current phase
        last_temp_avg = self.arms_phases[self.last_arm][AVG]
        last_temp_impressions = self.arms_phases[self.last_arm][CNT_TOTAL] - 1
        #print("last temp impressions of arm " + str(self.last_arm) + ": " + str(last_temp_impressions))
        #print(self.arms_phases[self.last_arm][CNT_TOTAL])
        new_temp_avg = (last_temp_avg*last_temp_impressions + reward) / (last_temp_impressions+1)
        self.arms_phases[self.last_arm][AVG] = new_temp_avg
        #print(self.arms_phases)


    def next_arm_to_explore2(self):
        for arm in range(self.num_arms):
            if arm not in self.inactive:
                if self.users_dict[self.last_user][arm][CNT_TOTAL] == 0 or self.arms_phases[arm][CNT_TOTAL] == 0:
                    return arm
        return EXPLORED_ALL_ARMS

    def check_risk_deactivation2(self):
        arms_to_activate = []
        sum_thresholds = np.sum(self.new_arms_thresh)
        if self.current_round % self.phase_len >= self.phase_len - sum_thresholds:
            for arm in range(self.num_arms):
                if arm not in self.inactive and self.arms_phases[arm][CNT_TOTAL] < self.new_arms_thresh[arm]:
                    arms_to_activate.append(arm)
        if not arms_to_activate:
            return EXPLORED_ALL_ARMS

        max_UCB = -1
        max_arm = -1
        for arm in arms_to_activate:
            if arm in self.inactive:
                continue
            UCB = self.users_dict[self.last_user][arm][AVG]
            UCB += np.sqrt(2 * np.log(self.num_rounds) / self.users_dict[self.last_user][arm][CNT_TOTAL])
            if UCB > max_UCB:
                max_UCB = UCB
                max_arm = arm
        return max_arm

    def deactivation2(self):
        """
        Those who need to get deactivated, becomes inactive. Initiallizes the count of the phase
        """
        for arm in range(self.num_arms):
            if arm not in self.inactive:
                if self.arms_phases[arm][CNT_TOTAL] < self.arms_thresh[arm]:
                    self.inactive.append(arm)
                    #print("The inactive: " + str(self.inactive) + ", in round" + str(self.current_round))
        if self.current_round >= (self.num_rounds/self.num_arms)**(2/3) * (np.log(self.num_rounds)**(1/3)):
            self.reassigning_thresholds2()
        self.arms_phases = np.zeros((self.num_arms, 2))
        #print("Round: " + str(self.current_round))

    def reassigning_thresholds2(self):
        # Checking at the end of a phase. We want to know the contribution of a current phase
        # to the thresholds, so we should KNOW HOW MUCH AN ARM EARNED IN AVG IN THE CURRENT PHASE
        utilities = []
        for arm in range(self.num_arms):
            if arm in self.inactive:
                utilities.append(0)
                continue
            # sum_expectation = 0
            # for user in range(self.num_users):
            #     sum_expectation += self.users_distribution[user]*self.users_dict[user][arm][AVG]
            utilities.append(self.arms_phases[arm][AVG])
            #print("Arm score " + str(self.arms_phases[arm][AVG]))
        #print("Utilities: " + str(utilities))
        utilities = list(utilities/np.sum(utilities)) # in [0,1]
        new_thresholds = []
        for arm in range(self.num_arms):
            if arm in self.inactive:
                new_thresholds.append(0)
                continue
            new_thresholds.append((2*self.new_arms_thresh[arm]/self.phase_len + utilities[arm])/3)
        self.new_arms_thresh = list((new_thresholds / np.sum(new_thresholds))*self.phase_len)





