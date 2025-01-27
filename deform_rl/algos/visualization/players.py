from pathlib import Path
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
# functions for import of hyperparameters and playing results
from ..training.training_helpers import single_env_maker, create_multi_env
import pygame

from typing import Callable


def play_model(model_path: str | Path, normalize_path: str, maker: Callable[[], gym.Env], normalize=True):
    """
    Play a model on the environment.

    :param model_path: (str) the path to the model
    :param normalize_path: (str) the path to the VecNormalize statistics
    :param maker: (callable) the function to create the environment
    """
    model = PPO.load(model_path, device='cpu')
    pygame.display.set_caption(model_path.name)
    # print(model.policy)
    env = create_multi_env(
        maker, 1, normalize_path=normalize_path, normalize=normalize)
    if normalize_path is not None:
        env.training = False
    else:
        print("No VecNormalize statistics found. Playing without normalization.")
    obs = env.reset()
    # print(obs.shape)
    cum_reward = cnt = 0
    episode_cnt = 0
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if normalize:
            cum_reward += VecNormalize.get_original_reward(env)
        else:
            cum_reward += reward
        cnt += 1
        if done:
            obs = env.reset()
            print("Episode done: ", cnt, "reward: ", cum_reward)
            cnt = 0
            cum_reward = 0
            episode_cnt += 1
        env.render()
        if pygame.event.get(pygame.QUIT):
            break

    env.close()
