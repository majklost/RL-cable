import inspect

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from pathlib import Path
from torch import nn

from deform_rl.algos.save_manager import get_paths, consistency_check, delete_experiment, forget_last_run, load_manager
from deform_rl.algos.training.training_helpers import *
from deform_rl.envs.Cable_obs_env.environment import *
from deform_rl.envs.sim.utils.seed_manager import init_manager

EXPERIMENTS_PATH = Path(__file__).parent.parent / "experiments"
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
load_manager(EXPERIMENTS_PATH)
BASE_NAME = 'cable-obs-'

"""
Default arch net_arch = dict(pi=[64, 64], vf=[64, 64])
activation nn.Tanh
"""


# random movement
# maker = single_env_maker(CableObsV0, wrappers=[TimeLimit, Monitor], wrappers_args=[
#     {'max_episode_steps': 1000}, {}], render_mode='human')
# env = maker()
# obs, info = env.reset()
# print(obs.shape)
# print(info)
# done = trunc = False
# while not (done or trunc):
#     action = env.action_space.sample()
#     obs, reward, done, trunc, info = env.step(action)
#     env.render()

def vanilla():
    env_name = CableObsV0.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=2000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        CableObsV0, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=2000000, callback=[ch_clb, eval_clb])
    print("Training done")


def vannila_bigger():
    env_name = CableObsV0.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=2000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        CableObsV0, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu', policy_kwargs=dict(
                    net_arch=dict(pi=[128, 256], vf=[128, 256]),
                    activation_fn=nn.Tanh,
                ))
    print("Training model")
    model.learn(total_timesteps=2000000, callback=[ch_clb, eval_clb])
    print("Training done")


def no_obs():
    env_name = EmptyObsV0.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=2000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        EmptyObsV0, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu', policy_kwargs=dict(
                    net_arch=dict(pi=[128, 256], vf=[128, 256]),
                    activation_fn=nn.Tanh,
                ))
    print("Training model")
    model.learn(total_timesteps=2000000, callback=[ch_clb, eval_clb])
    print("Training done")


def no_obs_small():
    env_name = EmptyObsV0.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=2000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        EmptyObsV0, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'], norm_paths=EXPERIMENTS_PATH / "norms" / "cable-obs-no_obs_no_rew" / "cable-obs-no_obs_no_rew_r1_30-01-21-37-18.pkl")
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    # model = PPO("MlpPolicy", env, verbose=0,
    #             tensorboard_log=paths['tb'], device='cpu')
    model = PPO.load(
        EXPERIMENTS_PATH / "models" / "cable-obs-no_obs_no_rew" / "cable-obs-no_obs_no_rew_best_r1_30-01-21-37-18_best_model.zip")
    model.set_env(env)
    model.tensorboard_log = paths['tb']
    print("Training model")
    model.learn(total_timesteps=2000000, callback=[ch_clb, eval_clb])
    print("Training done")


def no_obs_no_rew():
    env_name = EmptyNoRewardObs.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=2000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        EmptyNoRewardObs, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=2000000, callback=[ch_clb, eval_clb])
    print("Training done")


def no_obs_no_rew_bigger():
    env_name = EmptyNoRewardObs.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=2000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        EmptyNoRewardObs, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu', policy_kwargs=dict(
                    net_arch=dict(pi=[128, 256], vf=[128, 256]),
                    activation_fn=nn.Tanh,
                ))
    print("Training model")
    model.learn(total_timesteps=2000000, callback=[ch_clb, eval_clb])
    print("Training done")


def no_obs_no_rew_short_c():
    env_name = EmptyNoRewShorter.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=2000))
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(
        EmptyNoRewShorter, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=2000000, callback=[ch_clb, eval_clb])
    print("Training done")


def testing_inference():
    env_name = EmptyNoRewardObs.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=2000))
    paths = get_paths(get_name(
    ), 'Same as no_obs_no_rew but started from learned agent', env_name, data=kwargs)

    env, eval_env = standard_envs(
        EmptyNoRewardObs, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'], norm_paths=EXPERIMENTS_PATH / "norms" / "cable-obs-no_obs_no_rew" / "cable-obs-no_obs_no_rew_r1_30-01-21-37-18.pkl")
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO.load(
        EXPERIMENTS_PATH / "models" / "cable-obs-no_obs_no_rew" / "cable-obs-no_obs_no_rew_best_r1_30-01-21-37-18_best_model.zip")
    model.set_env(env)
    model.tensorboard_log = paths['tb']
    print("Training model")
    model.learn(total_timesteps=1000000, callback=[ch_clb, eval_clb])
    print("Training done")


def obs_observed_only():
    """
    Obstacles observed but no dens reward about them
    """
    env_name = ObsObservedOnly.__name__
    kwargs = dict(env_kwargs=dict(), maker_kwargs=dict(max_episode_steps=2000))
    paths = get_paths(get_name(
    ), 'started from learned agent adding observation without reward change', env_name, data=kwargs)

    env, eval_env = standard_envs(
        ObsObservedOnly, env_kwargs=kwargs['env_kwargs'], maker_kwargs=kwargs['maker_kwargs'], norm_paths=EXPERIMENTS_PATH / "norms" / "cable-obs-no_obs_no_rew" / "cable-obs-no_obs_no_rew_r1_30-01-21-37-18.pkl")
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO.load(
        EXPERIMENTS_PATH / "models" / "cable-obs-no_obs_no_rew" / "cable-obs-no_obs_no_rew_best_r1_30-01-21-37-18_best_model.zip")
    model.set_env(env)
    model.tensorboard_log = paths['tb']
    print("Training model")
    model.learn(total_timesteps=2000000, callback=[ch_clb, eval_clb])
    print("Training done")


def get_name():
    return BASE_NAME + str(inspect.stack()[1][3])


if __name__ == "__main__":
    pass
    # vanilla()
    # vannila_bigger()
    # no_obs()
    # no_obs_small()
    # testing_inference()
    # no_obs_no_rew()
    # obs_observed_only()
    # no_obs_no_rew_bigger()
    no_obs_no_rew_short_c()
