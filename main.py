import sys

import numpy as np

from algorithm.markov import Markov
from algorithm.monte_carlo import MonteCarlo
from algorithm.util import algorithm_init
from config.algorithm_config import AlgorithmConfig
from config.game_config import GameConfig
from game.lovebird import game_env_init

if __name__ == '__main__':
    np.set_printoptions(linewidth=400)
    sys.setrecursionlimit(10**6)

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
