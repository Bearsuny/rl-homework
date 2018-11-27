import sys

import numpy as np

# from algorithm.markov import Markov
# from algorithm.monte_carlo import MonteCarlo
# from algorithm.temporal_difference import TemporalDifference
# from algorithm.util import algorithm_init
# from config.algorithm_config import AlgorithmConfig
# from config.game_config import GameConfig
# from game.lovebird import game_env_init

from algorithm.pong.pong import pong

if __name__ == '__main__':
    # np.set_printoptions(linewidth=400)
    # sys.setrecursionlimit(10**6)

    # markov = Markov(*algorithm_init(),
    #                 AlgorithmConfig.actions,
    #                 AlgorithmConfig.gamma,
    #                 AlgorithmConfig.markov_mode_space)
    # game, *game_obj = game_env_init()
    # game.loop(*game_obj, policy=markov)

    # monte_carlo = MonteCarlo(*algorithm_init(),
    #                          AlgorithmConfig.actions,
    #                          AlgorithmConfig.gamma,
    #                          AlgorithmConfig.monte_mode_space,
    #                          AlgorithmConfig.monte_on_policy_epsilon)
    # game, *game_obj = game_env_init()
    # game.loop(*game_obj, monte_carlo)

    # temporal_difference = TemporalDifference(*algorithm_init(),
    #                                          AlgorithmConfig.actions,
    #                                          AlgorithmConfig.gamma,
    #                                          AlgorithmConfig.temporal_difference_alpha,
    #                                          AlgorithmConfig.temporal_difference_epsilon,
    #                                          AlgorithmConfig.temporal_mode_space)
    # game, *game_obj = game_env_init()
    # game.loop(*game_obj, temporal_difference)

    pong()
