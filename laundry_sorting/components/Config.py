import itertools


class Config:
    def __init__(self):
        self.num = 30

        # tuning
        self.model_list = ['Sarsa', 'QLearning', 'DQN']
        self.noi_list = [2000]  # [1000, 10000, 20000, 50000]
        self.nop_list = [50, 100, 500]
        self.reward_scale_list = [1, 2, 5]
        self.train_alpha_list = [0.1, 0.3, 0.5, 0.7]
        self.update_alpha_list = [0.1, 0.3, 0.5, 0.7]
        self.gamma_list = [0]  # [0, 0.3, 0.5, 0.7]
        self.epsilon_list = [0.1]  # [0.1, 0.2]
        self.correct_scale_list = [1, 2, 3, 5]  # [1, 2, 3, 5, 10]
        self.incorrect_scale_list = [0.5, 1]  # [0.5, 1, 2, 3]

        self.model_list = ['Sarsa', 'QLearning', 'DQN']
        self.noi_list = [2000]  # [1000, 10000, 20000, 50000]
        self.nop_list = [50]
        self.reward_scale_list = [1]
        self.train_alpha_list = [0.05]  # [0.05, 0.1, 0.3, 0.5, 0.7]
        self.update_alpha_list = [0.05, 0.1, 0.3, 0.5, 0.7]
        self.gamma_list = [0]  # [0, 0.3, 0.5, 0.7]
        self.epsilon_list = [0.1]  # [0.1, 0.2]
        self.correct_scale_list = [1, 2, 3, 5]  # [1, 2, 3, 5, 10]
        self.incorrect_scale_list = [0.5, 1, 2]  # [0.5, 1, 2, 3]

        # basic parameter
        self.baskets = {
            1: 'white',
            3: 'dark',
            5: 'colour'
        }

        self.img_dict = {
            0: 'og',
            1: 'ud',
            2: 'lr',
            3: 'affine',
            4: 'rot1',
            5: 'rot2',
            6: 'scale',
            7: 'blur',
            8: 'add',
            9: 'com1',
            10: 'com2',
            11: 'com3'
        }

        # parameter for DQN
        self.dqn_para = {
            'episode': 300,
            'state_dim': [None, 400, 300, 3],
            'img_size': (400, 300, 3),
            'action_dim': 3,
            'lr': 0.000001,
            'gamma': 0,
            'epsilon': 0.1,
            'batch_size': 32,
            'buffer_size': 500,
            'update_iter': 200,
            'start_learning': 50
        }

        # self.dqn_para = {
        #     'episode': 5,
        #     'state_dim': [None, 400, 300, 3],
        #     'img_size': (400, 300, 3),
        #     'action_dim': 3,
        #     'lr': 0.01,
        #     'gamma': 0,
        #     'epsilon': 0.1,
        #     'batch_size': 1,
        #     'buffer_size': 3,
        #     'update_iter': 5,
        #     'start_learning': 5
        # }

        # Model
        self.model = self.model_list[2]
        # TODO - gamma = 0?

        # Training
        self.noi = 2000
        self.reward_scale = 1
        self.train_alpha = 0.05
        self.train_gamma = 0
        self.train_epsilon = 0.1

        # Updating
        self.nop = 50  # number of training using a simple sorting behaviour
        self.update_alpha = 0.5
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
                                                   self.train_alpha_list,
                                                   self.gamma_list,
                                                   self.epsilon_list,
                                                   self.update_alpha_list,
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
