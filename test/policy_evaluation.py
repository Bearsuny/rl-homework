import numpy as np

from algorithm.markov import Markov
from config.algorithm_config import AlgorithmConfig


def test_policy_evaluation():
    reward_grid = -np.ones((4, 4), dtype=np.int)
    reward_grid[0][0] = 0
    reward_grid[3][3] = 0
    markov = Markov(reward_grid, [-1, 1, 0],
                    AlgorithmConfig.actions,
                    AlgorithmConfig.gamma,
                    AlgorithmConfig.markov_mode_space)
    markov.mode = markov.mode_space[0]
    markov.change_mode(markov.mode)
    print(markov.i_rewards)
    print(markov.v_values)
