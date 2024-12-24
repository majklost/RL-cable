from pathlib import Path

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from torch import nn
from deform_rl.algos.save_manager import get_paths, consistency_check, delete_experiment, load_manager
from deform_rl.algos.training.training_helpers import *
from deform_rl.envs.Rectangle_env.debug_env import *
from deform_rl.algos.lego.networks import *
# delete_experiment('rect2D')

EXPERIMENTS_PATH = Path(__file__).parent.parent.parent / \
    "experiments"/'debugging'
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
print(EXPERIMENTS_PATH)
load_manager(EXPERIMENTS_PATH)

BASE_NAME = 'debug-'

GAMMA = 0.99
TIMESTEPS = 600000


def noVel():
    env_name = RectNoVel.__name__
    NORMALIZE = True
    data = dict(env_kwargs=dict(gamma=GAMMA), normalize=NORMALIZE)
    paths = get_paths(get_name(BASE_NAME), 'first_run', env_name, data=data)
    env, eval_env = standard_envs(
        RectNoVel, data['env_kwargs'], normalize=NORMALIZE)
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu', gamma=GAMMA)
    print("Training model")
    model.learn(total_timesteps=TIMESTEPS, callback=[
                ch_clb, ev_clb])
    print("Training done")


def noNormNoVel():
    env_name = RectNoVel.__name__
    NORMALIZE = False
    data = dict(env_kwargs=dict(gamma=GAMMA), normalize=NORMALIZE)
    paths = get_paths(get_name(BASE_NAME), 'first_run', env_name, data=data)
    env, eval_env = standard_envs(
        RectNoVel, data['env_kwargs'], normalize=NORMALIZE)
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu', gamma=GAMMA)
    print("Training model")
    model.learn(total_timesteps=TIMESTEPS, callback=[
                ch_clb, ev_clb])
    print("Training done")


def vel():
    env_name = RectVel.__name__
    NORMALIZE = True
    data = dict(env_kwargs=dict(gamma=GAMMA), normalize=NORMALIZE)
    paths = get_paths(get_name(BASE_NAME), 'first_run', env_name, data=data)
    env, eval_env = standard_envs(
        RectVel, data['env_kwargs'], normalize=NORMALIZE)
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu', gamma=GAMMA)
    print("Training model")
    model.learn(total_timesteps=TIMESTEPS, callback=[
                ch_clb, ev_clb])
    print("Training done")


def noNormVel():
    env_name = RectVel.__name__
    NORMALIZE = False
    data = dict(env_kwargs=dict(gamma=GAMMA), normalize=NORMALIZE)
    paths = get_paths(get_name(BASE_NAME), 'first_run', env_name, data=data)
    env, eval_env = standard_envs(
        RectVel, data['env_kwargs'], normalize=NORMALIZE)
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu', gamma=GAMMA)
    print("Training model")
    model.learn(total_timesteps=TIMESTEPS, callback=[
                ch_clb, ev_clb])
    print("Training done")


def trajectoryVel():
    env_name = TrajectoryVel.__name__
    NORMALIZE = True
    data = dict(env_kwargs=dict(gamma=GAMMA), normalize=NORMALIZE)
    paths = get_paths(get_name(BASE_NAME), 'first_run', env_name, data=data)
    env, eval_env = standard_envs(
        TrajectoryVel, data['env_kwargs'], normalize=NORMALIZE)
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu', gamma=GAMMA)
    print("Training model")
    model.learn(total_timesteps=TIMESTEPS, callback=[
                ch_clb, ev_clb])
    print("Training done")


if __name__ == '__main__':
    # vel()
    # noVel()
    # noNormNoVel()
    # noNormVel()
    trajectoryVel()
