from gym import spaces
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

class StockTradingEnvironment():
    def __init__(self, csv_file):  # 初始化
        # 读取CSV文件
        self.data = pd.read_csv(csv_file)
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,))  # 状态空间维度
        self.action_space = spaces.Discrete(3)  # 动作空间维度
        self.current_step = 0  # 当前时间步
        self.trade_amount = 0
        self.cash_balance = 100000  # 初始现金余额
        self.stock_quantity = self.data.iloc[0]['open_oi']  # 初始股票持仓量
        self.stock_price = self.data.iloc[0]['open']  # 初始股票价格
        self.total_assets = self.cash_balance  # 初始总资产
        self.previous_cash_balance = 0  # 上一个时间步的现金余额
        self.previous_stock_quantity = 0  # 上一个时间步的股票数量
        self.previous_price = 0  # 上一个时间步的价格

    def reset(self):  # 重置环境到初始状态
        self.current_step = 0
        self.trade_amount = 0
        self.cash_balance = 100000
        self.stock_quantity = self.data.iloc[0]['open_oi']
        self.stock_price = self.data.iloc[0]['open']
        self.total_assets = self.cash_balance
        self.previous_cash_balance = 0  # 上一个时间步的现金余额
        self.previous_stock_quantity = 0  # 上一个时间步的股票数量
        self.previous_price = 0  # 上一个时间步的价格

        return self._get_observation()

    def step(self, action):  # 执行动作并观察下一个状态、奖励和是否结束
        self._take_action(action)
        self.current_step += 1
        reward = self._get_reward()
        done = self.current_step >= len(self.data) - 1
        return self._get_observation(), reward, done, {}

    def _get_observation(self):  # get_state  环境的观察值
        current_price = self.data.iloc[self.current_step]['close']
        observation = [current_price, self.stock_quantity, 0, 0, 0, 0]  # 将观测值扩展为长度为6 否则回报错
        return torch.tensor(observation, dtype=torch.float)  # 返回张量形式的观测值

    def _take_action(self, action):  # 动作空间
        trade_amount = 0
        current_price = self.data.iloc[self.current_step]['close']
        # print("current_price:  ", current_price)
        # print("self.stock_quantity:  ", self.stock_quantity)
        # print("trade_amount:  ", trade_amount)

        if action == 0:  # 卖出
            trade_amount = int(self.stock_quantity * 0.5)
            if trade_amount > 0:
                self.cash_balance += trade_amount * current_price  # 总资产
                self.stock_quantity -= trade_amount

        elif action == 1:  # 买入
            trade_amount = int(self.stock_quantity * 0.5)
            cash_balance = self.cash_balance
            if trade_amount > 0 and cash_balance > 0:  # 有钱才能买股票
                self.cash_balance -= trade_amount * current_price
                self.stock_quantity += trade_amount

        # print("action:  ", action)
        # print("trade_amount:  ", trade_amount)
        self.trade_amount = trade_amount

        # 更新上一个时间步的状态
        self.previous_cash_balance = self.cash_balance
        self.previous_stock_quantity = self.stock_quantity
        self.previous_price = current_price

    def _get_reward(self):
        current_price = self.data.iloc[self.current_step]['close']
        current_total_assets = self.cash_balance + self.stock_quantity * current_price
        previous_total_assets = self.previous_cash_balance + self.previous_stock_quantity * self.previous_price
        reward = current_total_assets - previous_total_assets
        return reward

# 定义DQN模型
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索因子
        self.epsilon_decay = 0.995  # 探索因子的衰减率
        self.epsilon_min = 0.01  # 探索因子的最小值
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.MSELoss()

    def decay_epsilon(self):  # epsilon=1时就是完全随机探索
        self.epsilon *= self.epsilon_decay

    def remember(self, state, action, reward, next_state, done):  #记忆回放
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        q_values = self.model(torch.FloatTensor(state))
        return torch.argmax(q_values).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_q_values = self.model(torch.FloatTensor(next_state))
                target = (reward + self.gamma * torch.max(next_q_values).item())
            q_values = self.model(torch.FloatTensor(state))
            target_q_values = q_values.clone()  # 目标固定
            target_q_values[action] = target   # 更新强化学习value
            loss = self.criterion(q_values, target_q_values.unsqueeze(0))
            self.optimizer.zero_grad()
            loss.backward()   # 更新神经网络
            self.optimizer.step()
            #  创新，随机探索概率。逐渐降低
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

env = StockTradingEnvironment('G:/000DT/code/TCL/data.csv')

agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)

rewards = []  # 记录每个训练周期的奖励值

for episode in range(1000):
    state = env.reset()
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state

        agent.replay(32)
        # print(reward)

    rewards.append(reward)
    print(reward)

    agent.decay_epsilon()

plt.plot(rewards, '-', c='r', label='reward')
plt.show()
