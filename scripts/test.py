
import pygame
import os
import numpy as np
from datetime import datetime

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from gymnasium.wrappers import FlattenObservation, TimeLimit, NormalizeObservation
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from deform_rl.envs.Rectangle_env.environment import Rectangle1D
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box, Dict


sim_cfg = {
    'width': 800,
    'height': 800,
    'FPS': 60,
    'gravity': 0,
    'damping': .15,
    'collision_slope': 0.01,
}


class CustomNormalizeObsrvation(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.width = env.width
        self.height = env.height
        max_length = np.linalg.norm([self.width, self.height])
        low = np.array([-self.width, -self.height, -np.inf, -np.inf])
        high = np.array([self.width, self.height, np.inf, np.inf])
        self.observation_space = Box(low=low, high=high, dtype=np.float32)

        # self.observation_space = Box(low=np.array([0.,0.,-1000.,-1000.,0.,0.]), high=np.array([800.,800.,1000.,1000.,800.,800.]), shape=(6,), dtype=np.float64)
        # self.observation_space = Box(low=0, high=1, shape=(4,), dtype=np.float64)
    # def observation(self, observation):
    #     mean = np.array([self.width, self.height]) / 2
    #     position = (observation['position'])
    #     velocity = observation['velocity']
    #     target = (observation['target'])
    # return np.concatenate([position, velocity, target])
    def observation(self, observation):
        # mean = np.array([self.width,self.height]) / 2
        position = observation['position']
        target = observation['target']
        velocity = observation['velocity']
        rel_target = target - position
        # rel_target /= np.array([self.width,self.height])
        # velocity /= np.array([self.width,self.height])

        return np.concatenate([rel_target, velocity], dtype=np.float32)


def _init(threshold=30, seed=None):
    # Base env
    env = Rectangle1D(sim_config=sim_cfg, threshold=threshold,
                      oneD=False, render_mode='human', seed=seed)
    env = CustomNormalizeObsrvation(env)
    # Apply wrappers
    # env = FlattenObservation(env)
    env = TimeLimit(env, max_episode_steps=1000)
    check_env(env, warn=True)

    return env


# random pick actions and visualize
tenv = _init()
obs, _ = tenv.reset(seed=24)
# tenv = eval_env
save_dir = "./saved_models"
t_model = PPO.load(os.path.join(save_dir, "best_model.zip"), force_reset=True)
cnt = 0
for i in range(10000):
    if cnt >= 1000:
        print("Killed by timeout")
        obs, _ = tenv.reset()
        cnt = 0
    action, _ = t_model.predict(obs, deterministic=True)
    obs, reward, done, truncated, info = tenv.step(action)
    tenv.render()
    if done:
        obs, _ = tenv.reset()
        print("Episode done: ", cnt)
        cnt = 0
    if pygame.event.get(pygame.QUIT):
        break
    cnt += 1
tenv.close()
