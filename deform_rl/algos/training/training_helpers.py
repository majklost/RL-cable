# basic training functions
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, EvalCallback
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from pathlib import Path
from typing import Callable
import inspect
import warnings


def single_env_maker(ENV_CREATOR: gym.Env, seed=0, wrappers: list[gym.Wrapper] = [],  wrappers_args: list[dict] = [], **kwargs):
    """
    Return a function that creates a single environment.

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param wrappers: (list) list of wrappers to use
    :param wrappers_args: (list) list of dictionaries with arguments for the wrappers
    :param kwargs: (dict) the arguments for the environment
    """
    call_num = 0
    assert len(wrappers) == len(
        wrappers_args), "The number of wrappers and their arguments must match"

    def _init():
        nonlocal call_num

        env = ENV_CREATOR(**kwargs)
        for wrapper, wrapper_args in zip(wrappers, wrappers_args):
            env = wrapper(env, **wrapper_args)
        env.reset(seed=seed + call_num)
        call_num += 1
        return env

    set_random_seed(seed)
    return _init


def create_multi_env(single_env_make: Callable[[], gym.Env], n_envs: int, normalize: bool = False, normalize_path: Path = None):
    """
    Create vectorized environment.

    :param single_env_make: (callable) a function that creates a single environment
    :param n_envs: (int) the number of environments to create
    """
    envs = DummyVecEnv([single_env_make for _ in range(n_envs)])
    if normalize and normalize_path is not None:
        envs = VecNormalize.load(normalize_path, envs)
    elif normalize:
        envs = VecNormalize(envs)
    else:
        envs = envs
    return envs


class SaveNormalizeCallback(BaseCallback):
    """
    Callback for saving VecNormalize statistics.
    :param save_path: (Path) the path to save the statistics
    :param save_freq: (int) the frequency at which to save VecNormalize statistics
    """

    def __init__(self, save_path: Path, save_freq: int = 1, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            if self.model.get_vec_normalize_env() is not None:
                self.model.get_vec_normalize_env().save(self.save_path)
        return True


class SaveModelCallback(BaseCallback):
    """
    Callback for saving the model. Not only best but also last.
    :param save_path: (Path) the path to save the model
    :param save_freq: (int) the frequency at which to save the model
    """

    def __init__(self, save_path: Path, save_freq: int = 1, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0:
            print(f"Saving model")
            self.model.save(self.save_path)
        return True


def get_name(base_name):
    return base_name + str(inspect.stack()[1][3])


def create_callback_list(paths, save_freq, eval_env) -> tuple:
    checkpoint_callback = CallbackList([SaveModelCallback(
        paths['model_last'], save_freq=save_freq), SaveNormalizeCallback(paths['norm'], save_freq=save_freq)])
    eval_callback = EvalCallback(
        eval_env=eval_env, eval_freq=save_freq, callback_on_new_best=SaveModelCallback(paths['model_best']))
    return checkpoint_callback, eval_callback


def standard_envs(env_cls, env_kwargs={}, n_train=4, n_eval=1, normalize=True, norm_paths=None):
    if env_kwargs == {}:
        warnings.warn("No environment arguments were provided")
    maker = single_env_maker(env_cls, wrappers=[TimeLimit, Monitor], wrappers_args=[
        {'max_episode_steps': 1000}, {}], render_mode='human', **env_kwargs)
    env = create_multi_env(maker, n_train, normalize=normalize,
                           normalize_path=norm_paths)
    eval_env = create_multi_env(
        maker, n_eval, normalize=normalize, normalize_path=norm_paths)
    return env, eval_env
