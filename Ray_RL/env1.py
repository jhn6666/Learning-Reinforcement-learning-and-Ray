import gym
import ray
from ray.rllib.algorithms import ppo

# 定义环境类
class CartPoleEnv(gym.Env):
    def __init__(self, env_config):
        self.env = gym.make("CartPole-v1")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        return self.env.step(action)

ray.init()

# 配置和初始化 PPO 算法
algo = ppo.PPO(env=CartPoleEnv, config={
    "env_config": {},  # config to pass to env class
})

# 训练代理
while True:
    print(algo.train())
