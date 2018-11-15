import numpy as np


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

    def init_q_values(self):
        # arbitrarily Q(s, a)
        self.q_values = np.zeros((self.i_rewards.shape[0], self.i_rewards.shape[1], len(self.actions)))

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
        # self.q_values = np.ones_like(self.q_values, dtype=np.float) * -1000
        self.return_values = np.zeros_like(self.return_values, dtype=np.int)
        self.return_values_count = np.zeros_like(self.return_values_count, dtype=np.int)
        self.pi_values = np.ones_like(self.return_values, dtype=np.float) * (1 / len(self.actions))

        if self.mode == 'exploring_starts':
            self.exploring_starts()
            # self.q_values = np.load('./output/05/q_values.npy').reshape(self.q_values.shape)
        if self.mode == 'on_policy':
            self.on_policy()
            # self.pi_values = np.load('./output/05/pi_values.npy').reshape(self.pi_values.shape)

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

    def generate_episode_recurrent(self, state):
        episode = []
        state_init = state.copy()
        while episode == [] or len(episode) > 1000:
            episode = []
            state = state_init.copy()
            next_state = state_init.copy()
            while not state in self.e_states:
                if self.mode == 'exploring_starts':
                    action_candidate = []
                    for action_id, action in enumerate(self.actions):
                        if self.q_values[state[0]][state[1]][self.actions.index(action)] == np.max(self.q_values[state[0]][state[1]]):
                            action_candidate.append(action)
                    action = np.random.choice(action_candidate)
                    # print(action, self.q_values[state[0]][state[1]])
                    # action = np.random.choice(self.actions)
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
                if len(episode) > 200:
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
        for i in range(100):
            print(f'{i} ...')
            episodes = self.generate_episodes(1)
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
