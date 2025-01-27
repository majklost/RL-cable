import os
import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from deform_rl.envs.Cable_reshape_env.environment import CableReshapeV2, CableReshape
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback

import pygame


def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = CableReshapeV2(render_mode='human', seg_num=10,
                             cable_length=300, scale_factor=800)
        env = TimeLimit(env, max_episode_steps=1000)
        env = Monitor(env)
        # use a seed for reproducibility
        # Important: use a different seed for each environment
        # otherwise they would generate the same experiences
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


env2 = DummyVecEnv([make_env(i+4) for i in range(1)])
save_dir = os.path.join("saved_models/reshape")
# log_dir = os.path.join("logs/reshape")
testing_env = VecNormalize.load(save_dir+"/vecnorms.pkl", env2)
testing_env.training = False
model = PPO.load(save_dir+"/best_model.zip", device='cpu')
obs = testing_env.reset()
cum_reward = cnt = 0
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, done, info = testing_env.step(action)
    cum_reward += reward
    cnt += 1
    if done:
        obs = testing_env.reset()
        print("Episode done: ", cnt, "reward: ", cum_reward)

        cnt = 0
        cum_reward = 0
    testing_env.render()
    if pygame.event.get(pygame.QUIT):
        break
    # print(obs, reward)
testing_env.close()
