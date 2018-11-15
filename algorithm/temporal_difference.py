import numpy as np


class TemporalDifference():
    def __init__(self, reward_grid, reward_category, actions, gamma, alpha, epsilon, mode_space):
        self.i_rewards = reward_grid

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
        self.alpha = alpha
        self.epsilon = epsilon

        self.mode_space = mode_space
        self.mode = None

        self.q_values = None

    def init_q_values(self):
        # arbitrarily Q(s, a)
        self.q_values = -np.random.random((self.i_rewards.shape[0], self.i_rewards.shape[1], len(self.actions)))

        # Q(terminal, *) = 0
        for e_state in self.e_states:
            for action_id, action in enumerate(self.actions):
                self.q_values[e_state[0]][e_state[1]][action_id] = 0
                
        # can't arrive
        for s_state in self.s_states:
            for action_id, action in enumerate(self.actions):
                if s_state == self.get_next_state(s_state, action):
                    self.q_values[s_state[0]][s_state[1]][action_id] = -1000
        for c_state in self.c_states:
            for action_id, action in enumerate(self.actions):
                self.q_values[c_state[0]][c_state[1]][action_id] = -1000

    def change_mode(self, mode):
        self.init_q_values()

        if self.mode == 'sarsa':
            self.sarsa()
        if self.mode == 'qlearning':
            self.qlearning()

    def sarsa(self):
        episode = 0
        while True:
            last_q_values = self.q_values.copy()
            for state in self.s_states:
                action = self.greedy(state)
                count = 0
                while True:
                    next_state = self.get_next_state(state, action)
                    if next_state in self.e_states:
                        break
                    next_action = self.greedy(next_state)
                    reward = self.i_rewards[state[0]][state[1]]
                    self.sarsa_update_q_values(state, action, reward, next_state, next_action)
                    state = next_state
                    action = next_action
                    count += 1
                    if count > 500:
                        break
            print(f'Episode: {episode}, {np.max(last_q_values-self.q_values)}')
            if (last_q_values - self.q_values < 1).all():
                break
            episode += 1

    def qlearning(self):
        episode = 0
        while True:
            last_q_values = self.q_values.copy()
            for state in self.s_states:
                count = 0
                while True:
                    action = self.greedy(state)
                    next_state = self.get_next_state(state, action)
                    if next_state in self.e_states:
                        break
                    reward = self.i_rewards[state[0]][state[1]]
                    self.qlearning_update_q_values(state, action, reward, next_state)
                    state = next_state
                    count += 1
                    if count > 500:
                        break
            print(f'Episode: {episode}, {np.max(last_q_values-self.q_values)}')
            if (last_q_values - self.q_values < 0.1).all():
                break
            episode += 1

    def give_action_advice(self, state):
        action_candidate = []
        for action_id, action in enumerate(self.actions):
            if action == self.actions[np.argmax(self.q_values[state[0]][state[1]])]:
                action_candidate.append(action)
        action = np.random.choice(action_candidate)
        print(state, action_candidate, action, self.q_values[state[0]][state[1]])
        return action

    def greedy(self, state):
        actions_probability = np.zeros_like(self.actions, dtype=np.float)

        for action_id, action in enumerate(self.actions):
            if self.q_values[state[0]][state[1]][self.actions.index(action)] == np.max(self.q_values[state[0]][state[1]]):
                actions_probability[action_id] = 1 - self.epsilon + self.epsilon / len(self.actions)
            else:
                actions_probability[action_id] = self.epsilon / len(self.actions)
        actions_probability = actions_probability / np.sum(actions_probability)
        return np.random.choice(self.actions, p=actions_probability)

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

    def sarsa_update_q_values(self, state, action, reward, next_state, next_action):
        old_q_values = self.q_values[state[0]][state[1]][self.actions.index(action)]
        next_q_values = self.q_values[next_state[0]][next_state[1]][self.actions.index(next_action)]
        self.q_values[state[0]][state[1]][self.actions.index(action)] = (1-self.alpha) * old_q_values + self.alpha * (reward + self.gamma * next_q_values)
    
    def qlearning_update_q_values(self, state, action, reward, next_state):
        old_q_values = self.q_values[state[0]][state[1]][self.actions.index(action)]
        next_q_values = np.max(self.q_values[next_state[0]][next_state[1]])
        self.q_values[state[0]][state[1]][self.actions.index(action)] = (1-self.alpha) * old_q_values + self.alpha * (reward + self.gamma * next_q_values)