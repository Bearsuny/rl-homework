import sys
import os

import pygame
from pygame.locals import *

import numpy as np
from collections import Iterable


class GameConfig:
    caption = 'lovebird-ff'
    window_size = (960, 540)
    grid_size = (60, 60)
    row = window_size[1] // grid_size[1]
    col = window_size[0] // grid_size[0]

    brick_size = (138, 793)
    bricks_pos = [5, 5, 10, 10]
    bricks_height = [4, 3, 3, 4]

    bird_size = (780, 690)
    female_bird_pos = col-1

    fps = 30
    save_number = 0
    loop_flag = False

    root_path = '.'
    resource_path = 'assets'
    save_path = 'output'

    reward_category = [-1, -5, 1]
    actions = ['e', 'w', 's', 'n']
    gamma = 1

    markov_mode_space = ['random', 'policy_iteration', 'value_iteration']
    markov_mode_count = 0

    monte_mode_space = ['exploring_starts', 'on_policy']
    monte_mode_count = 1

    monte_on_policy_epsilon = 0.1


class GameItem(pygame.sprite.Sprite):
    def __init__(self, source_path, size):
        super().__init__()
        self.image = pygame.image.load(source_path).convert_alpha()
        self.image = pygame.transform.scale(self.image, size)
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)

    def update_rect(self):
        self.rect = self.image.get_rect()

    def update_mask(self):
        self.mask = pygame.mask.from_surface(self.image)

    def blit(self, screen):
        screen.blit(self.image, self.rect)


class Grid():
    def __init__(self, color, width, mode):
        self.containers = []
        self.color = color
        self.width = width
        self.mode = mode
        if mode == 'rect' or mode == 'value':
            self.text_font = pygame.font.SysFont('times', 15)
            self.text_surface = self.text_font.render(str(0), True, (0, 0, 0))

    def blit(self, screen):
        if self.mode == 'line':
            for item in self.containers:
                pygame.draw.line(screen, self.color, *item, self.width)
        if self.mode == 'rect':
            for i, item in enumerate(self.containers):
                pygame.draw.rect(screen, self.color, item, self.width)
                self.text_surface = self.text_font.render(str(i), True, (0, 0, 0))
                pos = item[0]
                screen.blit(self.text_surface, (pos[0] + 5, pos[1] + 5))
        if self.mode == 'value':
            for i, row in enumerate(self.containers):
                for j, value in enumerate(row):
                    self.text_surface = self.text_font.render(str(round(float(value), 2)), True, (0, 0, 0))
                    pos = (j*60+5, (i+1)*60-20)
                    screen.blit(self.text_surface, pos)


class Brick(GameItem):
    def __init__(self, source_path, source_size, grid_size):
        size = (grid_size[0], source_size[1] * grid_size[1] // source_size[0])
        super().__init__(source_path, size)

    def chop_with_flip(self, height, flip_flag, location):
        self.image = pygame.transform.chop(self.image, [0, self.image.get_size()[1] - height, 0, height]).convert_alpha()
        if flip_flag:
            self.image = pygame.transform.flip(self.image, False, True).convert_alpha()
        self.update_rect()
        self.rect[0], self.rect[1] = location
        self.update_mask()


class Bird(GameItem):
    def __init__(self, source_path, source_size, gender, window_size):
        super().__init__(source_path, source_size)
        self.gender = gender
        if gender == 'female':
            self.image = pygame.transform.flip(self.image, True, False).convert_alpha()
            self.update_rect()
            self.rect[0] = window_size[0]-self.image.get_size()[0]
            self.update_mask()

    def update(self, step, direction):
        if direction == 's':  # Down
            self.rect.move_ip(0, step)
        if direction == 'e':  # Right
            self.rect.move_ip(step, 0)
        if direction == 'n':  # Up
            self.rect.move_ip(0, -step)
        if direction == 'w':  # Left
            self.rect.move_ip(-step, 0)

    def collision(self, step, bricks, window_size):
        directions = [(0, step), (step, 0), (0, -step), (-step, 0)]
        rects = [self.rect.move(item) for item in directions]
        last_rect = self.rect
        candidates = []
        for i, item in enumerate(rects):
            self.rect = item
            collides = pygame.sprite.spritecollide(self, bricks, False, pygame.sprite.collide_mask)
            if collides:
                pass
            else:
                if 0 <= self.rect[0] <= window_size[0]-self.image.get_size()[0] and 0 <= self.rect[1] <= window_size[1]-self.image.get_size()[1]:
                    candidates.append(directions[i])
        choice = np.random.choice(len(candidates))
        direction = directions.index(candidates[choice])
        self.rect = last_rect
        return direction

    def find_mate(self, bird):
        return True if pygame.sprite.collide_mask(self, bird) else False

    def reset(self):
        self.rect[0], self.rect[1] = 0, 0


class LoveBirdGame():
    def __init__(self, caption, window_size):
        pygame.init()
        pygame.display.set_caption(caption)
        self.screen = pygame.display.set_mode(window_size)
        self.clock = pygame.time.Clock()
        self.history_grid = Grid((255, 0, 0), 5, 'rect')
        self.v_values_grid = Grid((255, 0, 0), 0, 'value')

    def loop(self, bg, grid, bricks, birds, policy):
        while True:
            self.clock.tick(GameConfig.fps)

            male_bird, female_bird = [bird for bird in birds]
            pos = ((male_bird.rect[0], male_bird.rect[1]), GameConfig.grid_size)
            if pos not in self.history_grid.containers:
                self.history_grid.containers.append(pos)

            if GameConfig.loop_flag:
                # action = male_bird.collision(GameConfig.grid_size[0], bricks, GameConfig.window_size)

                state_x = male_bird.rect[1] // GameConfig.grid_size[1]
                state_y = male_bird.rect[0] // GameConfig.grid_size[0]
                action = policy.give_action_advice([state_x, state_y])
                # print([state_x, state_y], action)
                male_bird.update(GameConfig.grid_size[0], action)
                if male_bird.find_mate(female_bird):
                    GameConfig.loop_flag = not GameConfig.loop_flag

            for event in pygame.event.get():
                if event.type == KEYDOWN:
                    if event.key == K_RETURN:
                        GameConfig.loop_flag = not GameConfig.loop_flag
                        self.reset()
                        male_bird.reset()
                        # policy.mode = policy.mode_space[GameConfig.markov_mode_count]
                        # if GameConfig.markov_mode_count < len(GameConfig.markov_mode_space) - 1:
                        #     GameConfig.markov_mode_count += 1
                        # else:
                        #     GameConfig.markov_mode_count = 0
                        # policy.mode = policy.mode_space[1]
                        # policy.change_mode(policy.mode)
                        # self.v_values_grid.containers = policy.v_values
                        policy.mode = policy.mode_space[GameConfig.monte_mode_count]
                        if GameConfig.monte_mode_count < len(GameConfig.monte_mode_space) - 1:
                            GameConfig.monte_mode_count += 1
                        else:
                            GameConfig.monte_mode_count = 0
                        policy.change_mode(policy.mode)
                    if event.key == K_ESCAPE:
                        pygame.image.save(self.screen, os.path.join(GameConfig.root_path, GameConfig.save_path, f'{GameConfig.save_number}.png'))
                        GameConfig.save_number += 1
                elif event.type == QUIT:
                    pygame.quit()
                    sys.exit()

            self.blit(bg, grid, bricks, birds, self.history_grid, self.v_values_grid)
            pygame.display.update()

    def blit(self, *objs):
        for obj in objs:
            if isinstance(obj, Iterable):
                for item in obj:
                    item.blit(self.screen)
            else:
                obj.blit(self.screen)

    def reset(self):
        self.history_grid.containers = []
        self.v_values_grid.containers = []


class Markov():
    def __init__(self, reward_grid, reward_category, actions, gamma, mode_space):
        self.i_rewards = reward_grid  # immediate_reward

        self.s_states = []  # search
        self.c_states = []  # crash
        self.e_states = []  # end
        for i, row in enumerate(reward_grid):
            for j, reward in enumerate(row):
                if reward == reward_category[0]:
                    self.s_states.append([i, j])
                if reward == reward_category[1]:
                    self.c_states.append([i, j])
                if reward == reward_category[2]:
                    self.e_states.append([i, j])

        self.actions = actions
        self.gamma = gamma

        self.v_values = np.zeros_like(reward_grid, dtype=np.float)
        for state in self.c_states:
            self.v_values[state[0]][state[1]] = -10000

        self.pi_values = np.ones((reward_grid.shape[0], reward_grid.shape[1], len(actions)), dtype=np.int) * (1/len(actions))
        self.q_values = np.zeros_like(self.pi_values, dtype=np.float)

        self.mode_space = mode_space
        self.mode = None

    def change_mode(self, mode):
        self.v_values = np.zeros_like(self.v_values, dtype=np.float)
        for state in self.c_states:
            self.v_values[state[0]][state[1]] = -10000
        self.pi_values = np.ones(self.pi_values.shape, dtype=np.int) * (1/len(self.actions))
        self.q_values = np.zeros_like(self.q_values, dtype=np.float)

        if self.mode == 'random':
            self.policy_evaluation()
            print(self.v_values)
        if self.mode == 'policy_iteration':
            self.policy_iteration()
            print(self.v_values)
        if self.mode == 'value_iteration':
            self.value_iteration()
            print(np.around(self.v_values, decimals=2))

    def get_next_state(self, state, action):
        next_state = state.copy()
        if action == 'e':
            next_state[1] += 1
        if action == 'w':
            next_state[1] -= 1
        if action == 's':
            next_state[0] += 1
        if action == 'n':
            next_state[0] -= 1
        if not (next_state in self.s_states or next_state in self.e_states):
            next_state = state.copy()
        return next_state

    # pi(a|s)
    def get_pi_value(self, state, action):
        return self.pi_values[state[0]][state[1]][self.actions.index(action)]

    # v_pi(s)
    def state_value_function(self, state):
        value = 0
        for i, action in enumerate(self.actions):
            self.q_values[state[0]][state[1]][i] = self.action_value_function(state, action)
            value += self.get_pi_value(state, action) * self.q_values[state[0]][state[1]][i]
        return value

    # q_pi(s,a)
    def action_value_function(self, state, action):
        next_state = self.get_next_state(state, action)
        if next_state == state:
            return self.i_rewards[state[0]][state[1]] + self.gamma * self.v_values[state[0]][state[1]]
        else:
            return self.i_rewards[state[0]][state[1]] + self.gamma * self.v_values[next_state[0]][next_state[1]]

    def policy_evaluation(self):
        k = 0
        while True:
            last_v_values = self.v_values.copy()
            for i, state in enumerate(self.s_states):
                self.v_values[state[0]][state[1]] = self.state_value_function(state)
            k += 1
            last_v_values = np.around(last_v_values, decimals=0)
            self.v_values = np.around(self.v_values, decimals=0)
            if (last_v_values == self.v_values).all():
                break
        print(f'{k} iterations for policy evaluation in {self.mode} mode.')

    def policy_iteration(self):
        k = 0
        while True:
            last_pi_values = self.pi_values.copy()
            self.v_values = np.zeros_like(self.v_values, dtype=np.float)
            for state in self.c_states:
                self.v_values[state[0]][state[1]] = -10000
            self.policy_evaluation()
            for i, state in enumerate(self.s_states):
                q_value_slice = self.q_values[state[0]][state[1]]
                max_q_value = max(q_value_slice)
                for j, q_value in enumerate(q_value_slice):
                    if q_value == max_q_value:
                        self.pi_values[state[0]][state[1]][j] = 1 / list(q_value_slice).count(max_q_value)
                    else:
                        self.pi_values[state[0]][state[1]][j] = 0
            k += 1
            if (last_pi_values == self.pi_values).all():
                break
        print(f'{k} iterations for policy iteration in {self.mode} mode.')

    def value_iteration(self):
        k = 0
        while True:
            last_v_values = self.v_values.copy()
            for i, state in enumerate(self.s_states):
                value_candidate = []
                for i, action in enumerate(self.actions):
                    self.q_values[state[0]][state[1]][i] = self.action_value_function(state, action)
                    value_candidate.append(self.get_pi_value(state, action) * self.q_values[state[0]][state[1]][i])
                self.v_values[state[0]][state[1]] = max(value_candidate)
            k += 1
            if (last_v_values == self.v_values).all():
                break
        print(f'{k} iterations for value iteration in {self.mode} mode.')

    def give_action_advice(self, state):
        v_values_candidate = []
        for action in self.actions:
            next_state = self.get_next_state(state, action)
            v_values_candidate.append(self.v_values[next_state[0]][next_state[1]])
        action_candidate = []
        for i, v_value in enumerate(v_values_candidate):
            if v_value == max(v_values_candidate):
                action_candidate.append(i)
        action_candidate = np.array(action_candidate, dtype=np.int)
        # print(action_candidate)
        return self.actions[np.random.choice(action_candidate, 1)[0]]


class MonteCarlo():
    def __init__(self, reward_grid, reward_category, actions, gamma, mode_space, *args):
        self.i_rewards = reward_grid  # immediate_reward

        self.s_states = []  # search
        self.c_states = []  # crash
        self.e_states = []  # end
        for i, row in enumerate(reward_grid):
            for j, reward in enumerate(row):
                if reward == reward_category[0]:
                    self.s_states.append([i, j])
                if reward == reward_category[1]:
                    self.c_states.append([i, j])
                if reward == reward_category[2]:
                    self.e_states.append([i, j])

        self.actions = actions

        self.q_values = np.ones((reward_grid.shape[0], reward_grid.shape[1], len(actions)), dtype=np.float) * -1000
        self.return_values = np.zeros((reward_grid.shape[0], reward_grid.shape[1], len(actions)), dtype=np.int)
        self.return_values_count = np.zeros((reward_grid.shape[0], reward_grid.shape[1], len(actions)), dtype=np.int)

        self.pi_values = np.ones_like(self.return_values, dtype=np.float) * (1 / len(self.actions))

        self.gamma = gamma

        self.mode_space = mode_space
        self.mode = None

        self.epsilon = args[0]

    def change_mode(self, mode):
        self.q_values = np.ones_like(self.q_values, dtype=np.float) * -1000
        self.return_values = np.zeros_like(self.return_values, dtype=np.int)
        self.return_values_count = np.zeros_like(self.return_values_count, dtype=np.int)
        self.pi_values = np.ones_like(self.return_values, dtype=np.float) * (1 / len(self.actions))

        if self.mode == 'exploring_starts':
            # self.exploring_starts()
            self.q_values = np.load('./output/q_values.npy').reshape(self.q_values.shape)
        if self.mode == 'on_policy':
            # self.on_policy()
            self.pi_values = np.load('./output/pi_values.npy').reshape(self.pi_values.shape)


    # def generate_episode(self, state, episode):
    #     if state in self.e_states:
    #         return
    #     if self.mode == 'exploring_starts':
    #         action = np.random.choice(self.actions)
    #     elif self.mode == 'on_policy':
    #         action = np.random.choice(self.actions, p=self.pi_values[state[0]][state[1]])
    #     next_state = state.copy()
    #     if action == 'e':
    #         next_state[1] += 1
    #     if action == 'w':
    #         next_state[1] -= 1
    #     if action == 's':
    #         next_state[0] += 1
    #     if action == 'n':
    #         next_state[0] -= 1
    #     if not (next_state in self.s_states or next_state in self.e_states):
    #         next_state = state.copy()

    #     episode.append((state, action))
    #     return self.generate_episode(next_state, episode)

    def generate_episode_recurrent(self, state):
        episode = []
        state_init = state.copy()
        while episode == [] or len(episode) > 1000:
            episode = []
            state = state_init.copy()
            next_state = state_init.copy()
            while not state in self.e_states:
                if self.mode == 'exploring_starts':
                    action = np.random.choice(self.actions)
                elif self.mode == 'on_policy':
                    action = np.random.choice(self.actions, p=self.pi_values[state[0]][state[1]])
                if action == 'e':
                    next_state[1] += 1
                if action == 'w':
                    next_state[1] -= 1
                if action == 's':
                    next_state[0] += 1
                if action == 'n':
                    next_state[0] -= 1
                if not (next_state in self.s_states or next_state in self.e_states):
                    next_state = state.copy()
                episode.append((state, action))
                state = next_state.copy()
                if len(episode) > 1000:
                    break
        return episode

    def generate_episodes(self, num):
        episodes = []
        for i in range(num):
            for j, state in enumerate(self.s_states):
                episode = self.generate_episode_recurrent(state)
                print(f'{i} epoch, {j} state, {len(episode)}')
                episodes.append(episode)
        return episodes

    def calculate_g_first_visit(self, episodes):
        for k, episode in enumerate(episodes):
            # print(f'{k} Processing...')
            g = 0
            for i in range(len(episode)-1, -1, -1):
                value = episode[i]
                g = self.i_rewards[value[0][0]][value[0][1]] + g * self.gamma
                if episode.index(value) == i:
                    self.return_values[value[0][0]][value[0][1]][self.actions.index(value[1])] += g
                    self.return_values_count[value[0][0]][value[0][1]][self.actions.index(value[1])] += 1

    def exploring_starts(self):
        episodes = self.generate_episodes(100)
        self.calculate_g_first_visit(episodes)

        for i in range(self.q_values.shape[0]):
            for j in range(self.q_values.shape[1]):
                for k in range(self.q_values.shape[2]):
                    if self.return_values_count[i][j][k] != 0:
                        self.q_values[i][j][k] = self.return_values[i][j][k] / self.return_values_count[i][j][k]

        print(self.q_values)

    def on_policy(self):
        for c in range(1000):
            print(f'{c} ....')
            state = [0, 0]
            episode = self.generate_episode_recurrent(state)
            episodes = []
            episodes.append(episode)
            self.calculate_g_first_visit(episodes)

            for i in range(self.q_values.shape[0]):
                for j in range(self.q_values.shape[1]):
                    for k in range(self.q_values.shape[2]):
                        if self.return_values_count[i][j][k] != 0:
                            self.q_values[i][j][k] = self.return_values[i][j][k] / self.return_values_count[i][j][k]

            for i in range(self.pi_values.shape[0]):
                for j in range(self.pi_values.shape[1]):
                    for k, action in enumerate(self.actions):
                        if action == self.actions[np.argmax(self.q_values[i][j])]:
                            self.pi_values[i][j][k] = 1 - self.epsilon + self.epsilon / len(self.actions)
                        else:
                            self.pi_values[i][j][k] = self.epsilon / len(self.actions)

    def give_action_advice(self, state):
        if self.mode == 'exploring_starts':
            print(self.q_values[state[0]][state[1]], self.actions[np.argmax(self.q_values[state[0]][state[1]])])
            return self.actions[np.argmax(self.q_values[state[0]][state[1]])]
        elif self.mode == 'on_policy':
            print(self.pi_values[state[0]][state[1]], self.actions[np.argmax(self.pi_values[state[0]][state[1]])])
            return self.actions[np.argmax(self.pi_values[state[0]][state[1]])]


def game_env_init():
    game = LoveBirdGame(GameConfig.caption, GameConfig.window_size)

    bg = GameItem(os.path.join(GameConfig.root_path, GameConfig.resource_path, 'background.png'), GameConfig.window_size)

    grid = Grid((255, 255, 255), 1, 'line')
    h_points = [item for item in range(0, GameConfig.window_size[0], GameConfig.grid_size[0])]
    v_points = [item for item in range(0, GameConfig.window_size[1], GameConfig.grid_size[1])]
    h_lines = [((item, 0), (item, GameConfig.window_size[1])) for item in h_points]
    v_lines = [((0, item), (GameConfig.window_size[0], item)) for item in v_points]
    grid.containers += h_lines
    grid.containers += v_lines

    bricks = pygame.sprite.Group()
    for i, location_item, height_item in zip(range(4), GameConfig.bricks_pos, GameConfig.bricks_height):
        brick = Brick(os.path.join(GameConfig.root_path, GameConfig.resource_path, 'brick.png'), GameConfig.brick_size, GameConfig.grid_size)
        height = brick.image.get_size()[1] - height_item * GameConfig.grid_size[1]
        flip_flag = True
        location = [location_item * GameConfig.grid_size[0], 0]
        if i % 2 == 0:
            flip_flag = not flip_flag
            location[1] = GameConfig.window_size[1] - height_item * GameConfig.grid_size[1]
        brick.chop_with_flip(height, flip_flag, location)
        bricks.add(brick)

    birds = pygame.sprite.Group()
    for i in range(2):
        gender = 'male' if i % 2 == 0 else 'female'
        bird = Bird(os.path.join(GameConfig.root_path, GameConfig.resource_path, 'bird.png'), GameConfig.grid_size, gender, GameConfig.window_size)
        birds.add(bird)

    return game, bg, grid, bricks, birds


def markov_env_init():
    reward_category = GameConfig.reward_category
    i_rewards = np.ones((GameConfig.row, GameConfig.col), dtype=np.int) * reward_category[0]
    for i, pos, height in zip(range(len(GameConfig.bricks_pos)), GameConfig.bricks_pos, GameConfig.bricks_height):
        for j in range(height):
            if i % 2 == 0:
                i_rewards[GameConfig.row-j-1][pos] = reward_category[1]
            else:
                i_rewards[j][pos] = reward_category[1]
    i_rewards[0][GameConfig.female_bird_pos] = reward_category[2]
    return i_rewards, reward_category


def monte_carlo_env_init():
    return markov_env_init()


def test_policy_evaluation():
    reward_grid = -np.ones((4, 4), dtype=np.int)
    reward_grid[0][0] = 0
    reward_grid[3][3] = 0
    markov = Markov(reward_grid, [-1, 1, 0], GameConfig.actions, GameConfig.gamma, GameConfig.markov_mode_space)
    markov.mode = markov.mode_space[0]
    markov.change_mode(markov.mode)
    print(markov.i_rewards)
    print(markov.v_values)


if __name__ == '__main__':
    np.set_printoptions(linewidth=400)
    sys.setrecursionlimit(10**6)

    # markov = Markov(*markov_env_init(), GameConfig.actions, GameConfig.gamma, GameConfig.markov_mode_space)
    # game, *game_obj = game_env_init()
    # game.loop(*game_obj, policy=markov)

    monte_carlo = MonteCarlo(*monte_carlo_env_init(), GameConfig.actions, GameConfig.gamma, GameConfig.monte_mode_space, GameConfig.monte_on_policy_epsilon)
    game, *game_obj = game_env_init()
    game.loop(*game_obj, monte_carlo)

    # test_policy_evaluation()
