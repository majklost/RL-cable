import pygame
import os
import numpy as np
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from gymnasium.wrappers import FlattenObservation, TimeLimit, NormalizeObservation

from deform_rl.envs.Cable_reshape_env.environment import CableReshape


def make_env(seed=None):
    env = CableReshape(render_mode='human', seed=seed,
                       seg_num=4, cable_length=300, scale_factor=800)
    env = TimeLimit(env, max_episode_steps=1000)
    return env


save_dir = "./saved_models/cable_reshape"
log_dir = "./logs/cable_reshape"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


# random pick actions and visualize
tenv = make_env(25)
obs, _ = tenv.reset()
# tenv = eval_env

# t_model = model
t_model = PPO.load(os.path.join(save_dir, "best_model.zip"),
                   force_reset=True, device='cpu')
EP_CNT = 10
ep_cnt = 0
cnt = 0
rev_sum = 0
for i in range(10000):
    if cnt >= 1000:
        print("Killed by timeout")
        obs, _ = tenv.reset()
        cnt = 0
    action, _ = t_model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = tenv.step(action)
    rev_sum += reward
    tenv.render()
    if done:
        obs, _ = tenv.reset()
        ep_cnt += 1
        print("Episode done: ", cnt, "Reward: ", rev_sum)
        cnt = 0
        rev_sum = 0
        if ep_cnt >= EP_CNT:
            break
    if pygame.event.get(pygame.QUIT):
        break
    cnt += 1
tenv.close()
