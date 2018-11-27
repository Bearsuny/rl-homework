import os
from collections import deque, namedtuple
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gym
from algorithm.pong.wrapper import wrapper

# 超参
EPISODE = 50000                 # 训练轮次
REPLAY_MEMORY_SIZE = 100000     # 保留游戏记录的条数
REPLAY_INIT_SIZE = 10000        # 保留一定条数的记录后再开始训练
TARGET_SYNC = 1000              # 将policy网络的参数同步到target网络的步长
BATCH_SIZE = 32                 # 批次大小
LEARNING_RATE = 0.0001          # 网络的学习效率
EPSILON = 0.1                   # 贪心策略的参数
GAMMA = 0.9                     # 计算奖赏的折扣因子
# 是否使用CUDA
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 定义游戏记录的形式
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class DQN(nn.Module):
    '''
    DQN网络：3个卷积层+2个全连接层
    '''

    def __init__(self, in_channels, in_h, in_w, num_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(inplace=True)
        )
        self.conv_size = np.prod(self.conv(torch.zeros(1, in_channels, in_h, in_w)).size())
        self.fc = nn.Sequential(
            nn.Linear(self.conv_size, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        # 灰度值归一化
        x = x.float() / 255.0
        x = self.conv(x)
        return self.fc(x.view(x.size()[0], -1))


class ReplayMemory:
    def __init__(self, maxlen):
        # FIFO队列
        self.memory = deque(maxlen=maxlen)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def is_sample(self, tick):
        return False if tick < REPLAY_INIT_SIZE else True

    def sample(self, batch_size):
        idx = np.random.choice(len(self.memory)-batch_size)
        return list(self.memory)[idx:idx+batch_size]


def select_action(state, net, num_actions, epsilon):
    if np.random.uniform(0, 1) > epsilon:
        with torch.no_grad():
            return net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[np.random.choice(num_actions)]], device=DEVICE, dtype=torch.long)


def train_model(data, policy_net, target_net, optimizer, criterion):
    # 从data中提取信息
    states, actions, next_states, rewards = Transition(*zip(*data))
    states = torch.cat(states).to(DEVICE)
    actions = torch.cat(actions).to(DEVICE)
    next_states = torch.cat(next_states).to(DEVICE)
    rewards = torch.cat(rewards).to(DEVICE)

    # 根据data中的当前的state，计算Q值-policy net
    current_q_values = policy_net(states).gather(1, actions)
    # 根据data中的下一步state，计算Q值-target net
    with torch.no_grad():
        next_state_q_values = target_net(next_states).max(1)[0].view(BATCH_SIZE, 1).data
    expected_values = rewards.to(DEVICE) + GAMMA * next_state_q_values.to(DEVICE)

    # 计算损失
    loss = criterion(current_q_values, expected_values)
    optimizer.zero_grad()
    # 反向传播
    loss.backward()
    # 为了加快收敛，强制将policy net中的参数置为[-1, 1]
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def pong():
    # 乒乓游戏的环境
    env = gym.make('PongNoFrameskip-v4')
    env = wrapper(env)
    # 保存游戏界面
    env = gym.wrappers.Monitor(env, './video/07/', force=True, video_callable=lambda count: count % 100 == 0)

    # 乒乓游戏的动作空间
    env_action_meanings = env.unwrapped.get_action_meanings()
    print(env_action_meanings)

    # 记录游戏数据Buffer
    buffer = ReplayMemory(maxlen=REPLAY_MEMORY_SIZE)

    # DQN网络的输入Size
    print(env.observation_space.shape)

    # 定义用于训练的policy_net
    policy_net = DQN(*env.observation_space.shape, env.action_space.n).to(DEVICE)
    target_net = DQN(*env.observation_space.shape, env.action_space.n).to(DEVICE)

    # 定义优化器（梯度下降的方法）
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)

    # 定义损失函数
    criterion = nn.SmoothL1Loss()

    # 载入训练参数以实现air training
    if os.path.isfile('./output/07/policy_params.pkl'):
        print('Load policy parameters...')
        policy_net.load_state_dict(torch.load('./output/07/policy_params.pkl'))
    if os.path.isfile('./output/07/policy_params.pkl'):
        print('Load target parameters...')
        target_net.load_state_dict(torch.load('./output/07/target_params.pkl'))

    for i in range(EPISODE):
        # 每一轮次获取初始state
        last_state = torch.from_numpy(np.array(env.reset())).to(DEVICE).unsqueeze(0)
        # 清空replay buffer
        buffer.memory.clear()
        # 保存训练参数
        torch.save(policy_net.state_dict(), './output/07/policy_params.pkl')
        torch.save(target_net.state_dict(), './output/07/target_params.pkl')

        for tick in count():
            # 贪心算法进行动作决策
            action = int(select_action(last_state, policy_net, env.action_space.n, EPSILON))
            # 根据动作生成下一状态及奖赏
            env.render()
            observation, reward, done, info = env.step(action)
            next_state = torch.from_numpy(np.array(observation)).to(DEVICE).unsqueeze(0)
            buffer.push(last_state, torch.tensor(action).view(1, 1), next_state, torch.tensor(reward).view(1, 1))
            # 更新状态
            last_state = next_state
            # 训练policy net：保留一定条数的记录后再开始训练
            # 从buffer中采样
            if buffer.is_sample(tick):
                data = buffer.sample(BATCH_SIZE)
                train_model(data, policy_net, target_net, optimizer, criterion)
            # 同步target net：间隔一定条数的记录后再进行同步
            if tick % TARGET_SYNC == 0:
                target_net.load_state_dict(policy_net.state_dict())
            if done:
                break
    env.close()
