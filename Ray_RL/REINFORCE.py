import gym
import numpy as np
import torch
import matplotlib.pyplot as plt

l1 = 4  # 输入数据长度为4
l2 = 150  # 隐藏层为150
l3 = 2  # 输出是一个用于向左向右动作长度为2的向量

env = gym.make("CartPole-v0")

model = torch.nn.Sequential(
    torch.nn.Linear(l1, l2),
    torch.nn.LeakyReLU(),  # leakyReLU的意思是，不用relu激活，可以自己去掉试一试，效果会变差。
    torch.nn.Linear(l2, l3),
    torch.nn.Softmax(dim=0)  # 输出是一个动作的softmax概率分布
)

learning_rate = 0.009
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# state1 = env.reset()
# pred = model(torch.from_numpy(state1).float())  # 调用策略网络模型产生预测的动作概率
# action = np.random.choice(np.array([0, 1]), p=pred.data.numpy())  # 从策略网络产生的概率分布中抽样一个动作
# state2, reward, done, info = env.step(action)  # 采取动作并获得新的状态和奖励。info变量由环境产生，但与环境无关



def discount_rewards(rewards, gamma=0.99):
    lenr = len(rewards)
    disc_return = torch.pow(gamma, torch.arange(lenr).float()) * rewards  # 计算指数衰减奖励
    disc_return /= disc_return.max()  # 讲奖励归一化到[0，1]以提高数值稳定性
    return disc_return


def loss_fn(preds, r):  # 损失函数期望一个对所采取动作的动作概率数组和贴现奖励
    return -1 * torch.sum(r * torch.log(preds))  # 用于计算概率的对数，乘损失奖励，对其求和，然后对结果取反


MAX_DUR = 200
MAX_EPISODES = 500
gamma = 0.99
score = []  # 记录训练期间轮次长度的列表
expectation = 0.0
for episode in range(MAX_EPISODES):
    curr_state = env.reset()
    # env.render()
    done = False
    transitions = []  # 一系列状态，动作，奖励(但是我们忽略奖励)

    for t in range(MAX_DUR):
        act_prob = model(torch.from_numpy(curr_state).float())  # 获取动作概率
        action = np.random.choice(np.array([0, 1]), p=act_prob.data.numpy())  # 随机选取一个动作
        prev_state = curr_state
        curr_state, _, done, info = env.step(action)  # 在环境中采取动作
        transitions.append((prev_state, action, t + 1))  # 存储这个转换
        if done:  # 游戏失败则退出循环
            break

    ep_len = len(transitions)
    score.append(ep_len)  # 存储轮次时长
    print(ep_len)
    reward_batch = torch.Tensor([r for (s, a, r) in transitions]).flip(dims=(0,))  # 在单个张量中收集轮次的所有奖励
    disc_returns = discount_rewards(reward_batch)  # 计算衰减奖励
    state_batch = torch.Tensor([s for (s, a, r) in transitions])  # 在单个张量中收集轮次中的状态
    action_batch = torch.Tensor([a for (s, a, r) in transitions])  # 在单个张量中收集轮次中的动作
    pred_batch = model(state_batch)  # 重新计算轮次中所有状态的动作概率
    prob_batch = pred_batch.gather(dim=1, index=action_batch.long().view(-1, 1)).squeeze()  # 取与实际采取动作关联的动作概率的子集
    loss = loss_fn(prob_batch, disc_returns)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    env.render()


score = np.array(score)
# avg_score = running_mean(score, 50)

plt.figure(figsize=(10, 7))
plt.ylabel("Episode Duration", fontsize=22)
plt.xlabel("Training Epochs", fontsize=22)
plt.plot(score, color='green')
plt.show()
