class AlgorithmConfig:
    reward_category = [-1, -5, 1]
    actions = ['e', 'w', 's', 'n']
    gamma = 1

    markov_mode_space = ['random', 'policy_iteration', 'value_iteration']
    markov_mode_count = 0

    monte_mode_space = ['exploring_starts', 'on_policy']
    monte_mode_count = 0
    monte_on_policy_epsilon = 0.1

    temporal_mode_space = ['sarsa', 'qlearning']
    temporal_mode_count = 0
    temporal_difference_alpha = 0.1
    temporal_difference_epsilon = 0.1
