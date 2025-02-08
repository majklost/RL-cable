import inspect

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from pathlib import Path
from torch import nn

from deform_rl.algos.save_manager import get_paths, consistency_check, delete_experiment, forget_last_run, load_manager
from deform_rl.algos.training.training_helpers import *
from deform_rl.envs.Cable_radius_env.environment import *
from deform_rl.envs.sim.utils.seed_manager import init_manager

EXPERIMENTS_PATH = Path(__file__).parent.parent / "experiments"
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
load_manager(EXPERIMENTS_PATH)
BASE_NAME = 'cable-radius-'

"""
Default arch net_arch = dict(pi=[64, 64], vf=[64, 64])
activation nn.Tanh
"""


def empty():
    env_name = CableRadiusEmpty.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=1000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        CableRadiusEmpty, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=1000000, callback=[ch_clb, eval_clb])
    print("Training done")


def obs():
    env_name = CableRadiusNearestObs.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=1000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        CableRadiusNearestObs, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')

    # model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=paths['tb'], device='cpu',
    #             batch_size=256, gamma=0.95, learning_rate=1.9851635274160808e-05, clip_range=0.2, n_epochs=20, gae_lambda=0.98,
    #             policy_kwargs=dict(
    #                 net_arch=dict(pi=[256, 256], vf=[256, 256]),
    #                 activation_fn=nn.ReLU)
    #             )

    print("Training model")
    model.learn(total_timesteps=4000000, callback=[
                ch_clb, eval_clb, SuccessRateTracker()])
    print("Training done")


def obs_stronger():
    env_name = CableRadiusNearestStronger.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=1000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        CableRadiusNearestStronger, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')

    # model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=paths['tb'], device='cpu',
    #             batch_size=256, gamma=0.95, learning_rate=1.9851635274160808e-05, clip_range=0.2, n_epochs=20, gae_lambda=0.98,
    #             policy_kwargs=dict(
    #                 net_arch=dict(pi=[256, 256], vf=[256, 256]),
    #                 activation_fn=nn.ReLU)
    #             )

    print("Training model")
    model.learn(total_timesteps=4000000, callback=[
                ch_clb, eval_clb, SuccessRateTracker()])
    print("Training done")


def obs_pseudo():
    env_name = CableRadiusNearestObs.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=1000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        CableRadiusNearestObs, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=paths['tb'], device='cpu',
                batch_size=256, gamma=0.95, learning_rate=1.9851635274160808e-05, clip_range=0.2, n_epochs=20, gae_lambda=0.98,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                    activation_fn=nn.ReLU)
                )

    print("Training model")
    model.learn(total_timesteps=4000000, callback=[
                ch_clb, eval_clb, SuccessRateTracker()])
    print("Training done")


def obs_vel():
    env_name = CableRadiusObsVel.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=1000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        CableRadiusObsVel, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=paths['tb'], device='cpu',
                batch_size=256, gamma=0.95, learning_rate=1.9851635274160808e-05, clip_range=0.2, n_epochs=20, gae_lambda=0.98,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                    activation_fn=nn.ReLU)
                )

    print("Training model")
    model.learn(total_timesteps=4000000, callback=[
                ch_clb, eval_clb, SuccessRateTracker()])
    print("Training done")


def obs_vel_stronger():
    env_name = CableRadiusObsVelStronger.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=1000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        CableRadiusObsVelStronger, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=paths['tb'], device='cpu',
                batch_size=256, gamma=0.95, learning_rate=1.9851635274160808e-05, clip_range=0.2, n_epochs=20, gae_lambda=0.98,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                    activation_fn=nn.ReLU)
                )

    print("Training model")
    model.learn(total_timesteps=4000000, callback=[
                ch_clb, eval_clb])
    print("Training done")


def obs_vel_stronger_tuned():
    env_name = CableRadiusObsVelStronger.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=1000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        CableRadiusObsVelStronger, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'], n_train=32)
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)

    model = PPO("MlpPolicy", env, verbose=0, tensorboard_log=paths['tb'], device='cpu',
                batch_size=32, gamma=0.9999, learning_rate=7.134646320811716e-05, clip_range=0.4, n_epochs=4, gae_lambda=0.98,
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                    activation_fn=nn.Tanh),

                )

    print("Training model")
    model.learn(total_timesteps=10000000, callback=[
                ch_clb, eval_clb])
    print("Training done")


def get_name():
    return BASE_NAME + str(inspect.stack()[1][3])


if __name__ == "__main__":
    # empty()
    # obs_pseudo()
    # obs_stronger()
    # obs_vel()
    # obs_vel_stronger()
    obs_vel_stronger_tuned()
