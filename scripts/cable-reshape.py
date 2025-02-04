import inspect

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from pathlib import Path
from torch import nn

from deform_rl.algos.save_manager import get_paths, consistency_check, delete_experiment, forget_last_run, load_manager
from deform_rl.algos.training.training_helpers import *
from deform_rl.envs.Cable_reshape_env.environment import *
from deform_rl.envs.sim.utils.seed_manager import init_manager


# consistency_check()
# Multiple cable reshape tasks with slight changes in the environment.
# Each approach is a new function

EXPERIMENTS_PATH = Path(__file__).parent.parent / "experiments"
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
print(EXPERIMENTS_PATH)
load_manager(EXPERIMENTS_PATH)

BASE_NAME = 'cable-reshape-'

"""
Default arch net_arch = dict(pi=[64, 64], vf=[64, 64])
activation nn.Tanh
"""


def posOnly(continue_run=False):
    # init_manager(25, 25)
    env_name = CableReshapeV2.__name__
    kwargs = dict(env_kwargs=dict(
        seg_num=10, cable_length=300, scale_factor=800))
    paths = get_paths(get_name(), 'small cable', env_name,
                      continue_run, data=kwargs)
    # print(paths)
    maker = single_env_maker(CableReshapeV2, wrappers=[TimeLimit, Monitor], wrappers_args=[
                             {'max_episode_steps': 1000}, {}], render_mode='human', **(kwargs['env_kwargs']))

    if not continue_run:
        env = create_multi_env(maker, 4, normalize=True)
        eval_env = create_multi_env(maker, 1, normalize=True)
    else:
        env = create_multi_env(maker, 4, normalize=True,
                               normalize_path=paths['norm'])
        eval_env = create_multi_env(
            maker, 1, normalize=True, normalize_path=paths['norm'])
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    if not continue_run:
        model = PPO("MlpPolicy", env, verbose=0,
                    tensorboard_log=paths['tb'], device='cpu')
    else:
        model = PPO.load(paths['model_last'], env=env,
                         device='cpu', tensorboard_log=paths['tb'])
    print("Training model")
    # print(f"Training done for {env_name}")
    model.learn(total_timesteps=1000000, callback=[
                ch_clb, ev_clb], reset_num_timesteps=not continue_run)
    print("Training done")


def posOnlyTuned():
    tuned_params = {
        'n_steps': 512,
        'gamma': 0.99,
        'learning_rate': 0.00010575621210188617,
        'batch_size': 512,
        'clip_range': 0.2,
        'n_epochs': 20,
    }
    env_name = CableReshapeV2.__name__
    kwargs = dict(env_kwargs=dict(
        seg_num=10, cable_length=300, scale_factor=800))
    paths = get_paths(get_name(), 'small cable', env_name,
                      False, data=kwargs)
    maker = single_env_maker(CableReshapeV2, wrappers=[TimeLimit, Monitor], wrappers_args=[
        {'max_episode_steps': 1000}, {}], render_mode='human', **(kwargs['env_kwargs']))
    env = create_multi_env(maker, 4, normalize=True)
    eval_env = create_multi_env(maker, 1, normalize=True)

    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu', **tuned_params)
    print("Training model")
    model.learn(total_timesteps=1000000, callback=[
                ch_clb, ev_clb])
    print("Training done")


def posOnlyHarder10(continue_run=False):
    # init_manager(25, 25)
    env_name = CableReshapeHardFlips.__name__

    kwargs = dict(env_kwargs=dict(
        seg_num=10, cable_length=300, scale_factor=800))
    paths = get_paths(get_name(), 'bigger cable', env_name,
                      continue_run, data=kwargs)

    maker = single_env_maker(CableReshapeHardFlips, wrappers=[TimeLimit, Monitor], wrappers_args=[
                             {'max_episode_steps': 1000}, {}], render_mode='human', **(kwargs['env_kwargs']))

    if not continue_run:
        env = create_multi_env(maker, 4, normalize=True)
        eval_env = create_multi_env(maker, 1, normalize=True)
    else:
        env = create_multi_env(maker, 4, normalize=True,
                               normalize_path=paths['norm'])
        eval_env = create_multi_env(
            maker, 1, normalize=True, normalize_path=paths['norm'])
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    if not continue_run:
        model = PPO("MlpPolicy", env, verbose=0,
                    tensorboard_log=paths['tb'], device='cpu')
    else:
        model = PPO.load(paths['model_last'], env=env,
                         device='cpu', tensorboard_log=paths['tb'])
    print("Training model")
    # print(f"Training done for {env_name}")
    model.learn(total_timesteps=1000000, callback=[
                ch_clb, ev_clb], reset_num_timesteps=not continue_run)
    print("Training done")


def posOnlyBiggerCable(continue_run=False):
    env_name = CableReshapeV2.__name__
    kwargs = dict(env_kwargs=dict(
        seg_num=10, cable_length=300, scale_factor=800))
    paths = get_paths(get_name(), 'bigger cable', env_name,
                      data=kwargs, continue_run=continue_run)
    maker = single_env_maker(CableReshapeV2, wrappers=[TimeLimit, Monitor], wrappers_args=[
                             {'max_episode_steps': 1000}, {}], render_mode='human', **(kwargs['env_kwargs']))
    if not continue_run:
        env = create_multi_env(maker, 4, normalize=True)
        eval_env = create_multi_env(maker, 1, normalize=True)
    else:
        env = create_multi_env(maker, 4, normalize=True,
                               normalize_path=paths['norm'])
        eval_env = create_multi_env(
            maker, 1, normalize=True, normalize_path=paths['norm'])
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    if not continue_run:
        model = PPO("MlpPolicy", env, verbose=0,
                    tensorboard_log=paths['tb'], device='cpu')
    else:
        model = PPO.load(paths['model_last'], env=env,
                         device='cpu', tensorboard_log=paths['tb'])
        print("Training model")
    # print(f"Training done for {env_name}")
    model.learn(total_timesteps=1000000, callback=[
                ch_clb, ev_clb], reset_num_timesteps=not continue_run)
    print("Training done")


def posOnlyBiggerCable40(continue_run=False):
    env_name = CableReshapeV2.__name__
    kwargs = dict(env_kwargs=dict(
        seg_num=40, cable_length=300, scale_factor=800))
    paths = get_paths(get_name(), 'bigger cable', env_name,
                      data=kwargs, continue_run=continue_run)
    if not continue_run:
        env = create_multi_env(maker, 4, normalize=True)
        eval_env = create_multi_env(maker, 1, normalize=True)
    else:
        maker = single_env_maker(CableReshapeV2, wrappers=[TimeLimit, Monitor], wrappers_args=[
            {'max_episode_steps': 1000}, {}], render_mode='human', **(kwargs['env_kwargs']))
        env = create_multi_env(maker, 4, normalize=True,
                               normalize_path=paths['norm'])
        eval_env = create_multi_env(
            maker, 1, normalize=True, normalize_path=paths['norm'])
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    if not continue_run:
        model = PPO("MlpPolicy", env, verbose=0,
                    tensorboard_log=paths['tb'], device='cpu')
    else:
        model = PPO.load(paths['model_last'], env=env,
                         device='cpu', tensorboard_log=paths['tb'])
        print("Training model")
    # print(f"Training done for {env_name}")
    model.learn(total_timesteps=1000000, callback=[
                ch_clb, ev_clb], reset_num_timesteps=not continue_run)
    print("Training done")


def movementCable10():
    env_name = CableReshapeMovementOld.__name__
    kwargs = dict(env_kwargs=dict(
        seg_num=10, cable_length=300, scale_factor=800))
    paths = get_paths(get_name(), 'movement cable', env_name,
                      data=kwargs, continue_run=False)
    env, eval_env = standard_envs(
        CableReshapeMovementOld, env_kwargs=kwargs['env_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=1000000, callback=[
                ch_clb, ev_clb])
    print("Training done")


def movementCable10smallerThresh():
    env_name = CableReshapeMovementOld.__name__
    kwargs = dict(env_kwargs=dict(
        seg_num=10, cable_length=300, scale_factor=800, threshold=10))
    paths = get_paths(get_name(), 'movement cable', env_name,
                      data=kwargs, continue_run=False)
    env, eval_env = standard_envs(
        CableReshapeMovementOld, env_kwargs=kwargs['env_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=2000000, callback=[
                ch_clb, ev_clb])
    print("Training done")


def movementCable10Tuned():
    """
    Used tuned from posOnlyTuned
    """
    tuned_params = {
        'n_steps': 512,
        'gamma': 0.99,
        'learning_rate': 0.00026650142315084497,
        'batch_size': 256,
        'clip_range': 0.1,
        'n_epochs': 4,
        'gae_lambda': 0.95,
        'policy_kwargs': dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.ReLU,
        )
    }
    env_name = CableReshapeMovementOld.__name__
    kwargs = dict(env_kwargs=dict(
        seg_num=10, cable_length=300, scale_factor=800))
    paths = get_paths(get_name(), 'movement cable', env_name,
                      data=kwargs, continue_run=False)
    env, eval_env = standard_envs(
        CableReshapeMovementOld, env_kwargs=kwargs['env_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu', **tuned_params)
    print("Training model")
    model.learn(total_timesteps=1000000, callback=[
                ch_clb, ev_clb])
    print("Training done")


def reshapeNeighbour():
    """
    Reshape where there is information about the neighbours
    """
    env_name = CableReshapeNeighbourObs.__name__
    kwargs = dict(env_kwargs=dict(
        seg_num=10, cable_length=300, scale_factor=800))
    paths = get_paths(get_name(), 'neighbour obs', env_name,
                      data=kwargs, continue_run=False)
    env, eval_env = standard_envs(
        CableReshapeNeighbourObs, env_kwargs=kwargs['env_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=1000000, callback=[
                ch_clb, ev_clb])
    print("Training done")


def reshapePotentialRewardPosOnly():
    """
    Reshape with potential reward
    """
    env_name = CableReshapeV3.__name__
    kwargs = dict(env_kwargs=dict(
        seg_num=10, cable_length=300, scale_factor=800))
    paths = get_paths(get_name(), 'potential reward', env_name,
                      data=kwargs, continue_run=False)
    env, eval_env = standard_envs(
        CableReshapeV3, env_kwargs=kwargs['env_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=1000000, callback=[
                ch_clb, ev_clb])
    print("Training done")


def reshapePotentialRewardMovement():
    """
    Reshape with potential reward
    """
    env_name = CableReshapeMovement.__name__
    kwargs = dict(env_kwargs=dict(
        seg_num=10, cable_length=300, scale_factor=800))
    paths = get_paths(get_name(), 'potential reward', env_name,
                      data=kwargs, continue_run=False)
    env, eval_env = standard_envs(
        CableReshapeMovement, env_kwargs=kwargs['env_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=1000000, callback=[
                ch_clb, ev_clb])
    print("Training done")


def PotentialMovementVel():
    env_name = CableReshapeMovementVel.__name__
    kwargs = dict(env_kwargs=dict(
        seg_num=10, cable_length=300, scale_factor=800))
    paths = get_paths(get_name(), 'potential reward', env_name,
                      data=kwargs, continue_run=False)
    env, eval_env = standard_envs(
        CableReshapeMovementVel, env_kwargs=kwargs['env_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu',
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                    activation_fn=nn.ReLU,
                ))
    print("Training model")
    model.learn(total_timesteps=1500000, callback=[
                ch_clb, ev_clb])
    print("Training done")


def PotentialMovementNeighbourVel():
    env_name = CableReshapeMovementNeighbourObs.__name__
    kwargs = dict(env_kwargs=dict(
        seg_num=10, cable_length=300, scale_factor=800))
    paths = get_paths(get_name(), 'potential reward', env_name,
                      data=kwargs, continue_run=False)
    env, eval_env = standard_envs(
        CableReshapeMovementNeighbourObs, env_kwargs=kwargs['env_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu',
                policy_kwargs=dict(
                    net_arch=dict(pi=[256, 256], vf=[256, 256]),
                    activation_fn=nn.ReLU,
                ))
    print("Training model")
    model.learn(total_timesteps=1500000, callback=[
                ch_clb, ev_clb])
    print("Training done")


# HELPERS


def create_callback_list(paths, save_freq, eval_env) -> tuple:
    checkpoint_callback = CallbackList([SaveModelCallback(
        paths['model_last'], save_freq=save_freq), SaveNormalizeCallback(paths['norm'], save_freq=save_freq)])
    eval_callback = EvalCallback(
        eval_env=eval_env, eval_freq=save_freq, callback_on_new_best=SaveModelCallback(paths['model_best']))
    return checkpoint_callback, eval_callback


def get_name():
    return BASE_NAME + str(inspect.stack()[1][3])


if __name__ == "__main__":
    pass
    # posOnly(continue_run=False)
    # posOnlyTuned()
    # posOnly()
    # movementCable10()
    # forget_last_run('cable-reshape-posOnlyBiggerCable40')
    # posOnlyHarder10(continue_run=True)
    # posOnlyBiggerCable(continue_run=False)
    # posOnlyBiggerCable40(continue_run=True)
    # movementCable10Tuned()
    # movementCable10smallerThresh()
    # delete_experiment(BASE_NAME+'reshapeNeighbour')
    # reshapeNeighbour()
    # reshapePotentialRewardPosOnly()
    # reshapePotentialRewardMovement()
    PotentialMovementVel()
    # PotentialMovementNeighbourVel()
