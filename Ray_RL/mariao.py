import paddle
import paddle.nn as nn
import paddle.vision.transforms as T
import numpy as np
import random, datetime, os, copy

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# Super Mario environment for OpenAI Gym
import gym_super_mario_bros
import paddle.nn.functional as F
import collections

"""
request.txt
protobuf                           3.20.3
numpy                              1.19.0
gym                                0.18.0
gym-super-mario-bros               7.4.0
paddlepaddle                       2.0.2
"""

env = gym_super_mario_bros.make("SuperMarioBros-1-1-v0")

# Limit the action-space to
#   0. walk right
#   1. jump right
env = JoypadSpace(env, [["right"], ["right", "A"]])

env.reset()
next_state, reward, done, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")


class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transform = T.Grayscale()
        observation = transform(observation)
        observation = np.transpose(observation, (2, 0, 1)).squeeze(0)
        # observation = paddle.to_tensor(observation.copy(), dtype="float32")
        return observation


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose(
            [T.Resize(self.shape), T.Normalize(0, 255, data_format='HWC')]
            # [T.Resize(self.shape), T.Normalize(0, 255)] T.Normalize(mean=0, std=255, data_format='HWC')
        )

        observation = transforms(observation)

        return observation

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = ResizeObservation(env, shape=84)
env = GrayScaleObservation(env)
env = FrameStack(env, num_stack=4)

env.reset()
next_state, reward, done, info = env.step(action=0)
print(f"{next_state.shape},\n {reward},\n {done},\n {info}")
print(next_state)

class Model(nn.Layer):
    def __init__(self, num_inputs, num_actions):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2D(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2D(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2D(64, 64, 3, stride=1)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3136, 512)
        self.fc = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = self.linear(x)
        return self.fc(x)



class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, exp):
        self.buffer.append(exp)

    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        batch_obs, batch_action, batch_reword, batch_next_obs, batch_done = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, isOver = experience
            batch_obs.append(s)
            batch_action.append(a)
            batch_reword.append(r)
            batch_next_obs.append(s_p)
            batch_done.append(isOver)
        batch_obs = paddle.to_tensor(batch_obs, dtype='float32')
        batch_action = paddle.to_tensor(batch_action, dtype='int64')
        batch_reword = paddle.to_tensor(batch_reword, dtype='float32')
        batch_next_obs = paddle.to_tensor(batch_next_obs, dtype='float32')
        batch_done = paddle.to_tensor(batch_done, dtype='int64')

        return batch_obs, batch_action, batch_reword, batch_next_obs, batch_done

    def __len__(self):
        return len(self.buffer)



# 定义训练的参数
batch_size = 32  # batch大小
num_episodes = 10000  # 训练次数
memory_size = 20000  # 内存记忆
learning_rate = 1e-4  # 学习率大小
e_greed = 0.1  # 探索初始概率
gamma = 0.99  # 奖励系数
e_greed_decrement = 1e-6  # 在训练过程中，降低探索的概率
update_num = 0  # 用于计算目标模型更新次数
obs_shape = (4, 84, 84)  # 观测图像的维度
save_model_path = "models/model(1-1).pdparams"  # 保存模型路径


obs_dim = obs_shape[0]
action_dim = env.action_space.n

policyQ = Model(obs_dim, action_dim)
targetQ = Model(obs_dim, action_dim)
targetQ.eval()

if os.path.exists(save_model_path):
    model_state_dict  = paddle.load(save_model_path)
    policyQ.set_state_dict(model_state_dict )
    print('policyQ Model loaded')
    targetQ.set_state_dict(model_state_dict )
    print('targetQ Model loaded')

rpm = ReplayMemory(memory_size)
optimizer = paddle.optimizer.Adam(parameters=policyQ.parameters(),
                                  learning_rate=learning_rate)


# 评估模型
def evaluate():
    total_reward = 0
    obs = env.reset()
    while True:
        obs = np.expand_dims(obs, axis=0)
        obs = paddle.to_tensor(obs, dtype='float32')
        action = targetQ(obs)
        action = paddle.argmax(action).numpy()[0]
        next_obs, reward, done, info = env.step(action)
        obs = next_obs
        total_reward += reward

        if done:
            break
    return total_reward



def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.set_value( target_param * (1.0 - tau) + param * tau)

# 训练模型
def train():
    global e_greed, update_num
    total_reward = 0
    # 重置游戏状态
    obs = env.reset()

    while True:
        # 使用贪心策略获取游戏动作的来源
        e_greed = max(0.01, e_greed - e_greed_decrement)
        if np.random.rand() < e_greed:
            # 随机生成动作
            action = np.random.randint(action_dim)
        else:
            # 策略模型预测游戏动作
            obs1 = np.expand_dims(obs, axis=0)
            action = policyQ(paddle.to_tensor(obs1, dtype='float32'))
            action = paddle.argmax(action).numpy()[0]
        # 执行游戏
        next_obs, reward, done, info = env.step(action)
        env.render()
        total_reward += reward
        # 记录游戏数据
        rpm.append((obs, action, reward, next_obs, done))
        obs = next_obs
        # 游戏结束l
        if done:
            break
        # 记录的数据打印batch_size就开始训练
        if len(rpm) > batch_size:
            # 获取训练数据
            batch_obs, batch_action, batch_reword, batch_next_obs, batch_done = rpm.sample(batch_size)
            # 计算损失函数
            action_value = policyQ(batch_obs)
            action_onehot = paddle.nn.functional.one_hot(batch_action, action_dim)
            pred_action_value = paddle.sum(action_value * action_onehot, axis=1)

            batch_argmax_action = paddle.argmax(policyQ(batch_next_obs), axis=1)
            v = targetQ(batch_next_obs)
            select_v = []
            for i in range(v.shape[0]):
                select_v.append(v[i][int(batch_argmax_action[i].numpy()[0])])
            select_v = paddle.stack(select_v).squeeze()

            select_v.stop_gradient = True
            target = batch_reword + gamma * select_v * (1.0 - batch_done)

            cost = paddle.nn.functional.mse_loss(pred_action_value, target)
            # 梯度更新
            cost.backward()
            optimizer.step()
            optimizer.clear_grad()
            # 指定的训练次数更新一次目标模型的参数
            # if update_num % 200 == 0:
            #     targetQ.load_dict(policyQ.state_dict())
            # 软更新目标模型的参数
            soft_update(targetQ, policyQ, tau = 0.001)
            update_num += 1
    return total_reward


if __name__ == '__main__':
    episode = 0
    while episode < num_episodes:
        for t in range(3):
            train_reward = train()
            episode += 1
            print('Episode: {}, Reward: {:.2f}, e_greed: {:.2f}'.format(episode, train_reward, e_greed))

        if episode % 3 == 0:
            eval_reward = evaluate()
            print('Episode:{}    test_reward:{}'.format(episode, eval_reward))
            if eval_reward > 2500:
                paddle.save(targetQ.state_dict(), 'models/model(1-1)_test_{:.2f}.pdparams'.format(eval_reward))
        # 保存模型
        if not os.path.exists(os.path.dirname(save_model_path)):
            os.makedirs(os.path.dirname(save_model_path))
        paddle.save(targetQ.state_dict(), save_model_path)
