import numpy as np

from config.game_config import GameConfig


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
