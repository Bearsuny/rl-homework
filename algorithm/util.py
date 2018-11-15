import numpy as np

from config.algorithm_config import AlgorithmConfig
from config.game_config import GameConfig


def algorithm_init():
    reward_category = AlgorithmConfig.reward_category
    i_rewards = np.ones((GameConfig.row, GameConfig.col), dtype=np.int) * reward_category[0]
    for i, pos, height in zip(range(len(GameConfig.bricks_pos)), GameConfig.bricks_pos, GameConfig.bricks_height):
        for j in range(height):
            if i % 2 == 0:
                i_rewards[GameConfig.row-j-1][pos] = reward_category[1]
            else:
                i_rewards[j][pos] = reward_category[1]
    i_rewards[0][GameConfig.female_bird_pos] = reward_category[2]
    return i_rewards, reward_category
