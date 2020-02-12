import itertools


class Config:
    def __init__(self):
        self.num = 30

        self.model_list = ['Sarsa', 'QLearning']
        self.noi_list = [2000]  # [1000, 10000, 20000, 50000]
        self.nop_list = [50, 100, 500]
        self.reward_scale_list = [1, 2, 5]
        self.alpha_list = [0.1, 0.3, 0.5, 0.7]
        self.gamma_list = [0]  # [0, 0.3, 0.5, 0.7]
        self.epsilon_list = [0.1]  # [0.1, 0.2]
        self.correct_scale_list = [1, 2, 3, 5]  # [1, 2, 3, 5, 10]
        self.incorrect_scale_list = [0.5, 1]  # [0.5, 1, 2, 3]

        self.model_list = ['Sarsa', 'QLearning']
        self.noi_list = [2000]  # [1000, 10000, 20000, 50000]
        self.nop_list = [50, 100]
        self.reward_scale_list = [1, 2]
        self.alpha_list = [0.3, 0.5, 0.7]
        self.gamma_list = [0]  # [0, 0.3, 0.5, 0.7]
        self.epsilon_list = [0.1]  # [0.1, 0.2]
        self.correct_scale_list = [1, 3, 5]  # [1, 2, 3, 5, 10]
        self.incorrect_scale_list = [0.5, 1]  # [0.5, 1, 2, 3]

        # Model
        self.model = self.model_list[0]
        # TODO - gamma = 0?

        # Training
        self.noi = 2000
        self.reward_scale = 1
        self.train_alpha = 0.1
        self.train_gamma = 0
        self.train_epsilon = 0.1

        # Updating
        self.nop = 50  # number of training using a simple sorting behaviour
        self.update_alpha = 0.3
        self.update_gamma = 0
        self.update_epsilon = 0.1
        self.correct_scale = 3
        self.incorrect_scale = 1
        self.correct_reward = self.reward_scale * self.correct_scale
        self.incorrect_reward = self.reward_scale * self.incorrect_scale

        # parameter combinations
        self.combinations = list(itertools.product(self.model_list,
                                                   self.noi_list,
                                                   self.nop_list,
                                                   self.reward_scale_list,
                                                   self.alpha_list,
                                                   self.gamma_list,
                                                   self.epsilon_list,
                                                   self.alpha_list,
                                                   self.gamma_list,
                                                   self.epsilon_list,
                                                   self.correct_scale_list,
                                                   self.incorrect_scale_list))

        self.number_of_combinations = len(self.combinations)

    def set_parameters(self, parameter_list):
        self.model = parameter_list[0]
        self.noi = parameter_list[1]
        self.nop = parameter_list[2]
        self.reward_scale = parameter_list[3]
        self.train_alpha = parameter_list[4]
        self.train_gamma = parameter_list[5]
        self.train_epsilon = parameter_list[6]

        # Updating
        self.update_alpha = parameter_list[7]
        self.update_gamma = parameter_list[8]
        self.update_epsilon = parameter_list[9]
        self.correct_scale = parameter_list[10]
        self.incorrect_scale = parameter_list[11]
        self.correct_reward = self.reward_scale * self.correct_scale
        self.incorrect_reward = self.reward_scale * self.incorrect_scale
