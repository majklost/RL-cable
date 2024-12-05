import inspect

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


from deform_rl.algos.save_manager import get_paths, consistency_check, delete_experiment, forget_last_run
from deform_rl.algos.training.training_helpers import single_env_maker, create_multi_env, SaveNormalizeCallback, SaveModelCallback
from deform_rl.envs.Cable_reshape_env.environment import *
from deform_rl.envs.sim.utils.seed_manager import init_manager


consistency_check()
# Multiple cable reshape tasks with slight changes in the environment.
# Each approach is a new function


def posOnly(continue_run=False):
    # init_manager(25, 25)
    env_name = CableReshapeV2.__name__
    paths = get_paths('cable-reshape-'+str(inspect.stack()[0][3]), 'bigger cable', env_name,
                      continue_run)
    maker = single_env_maker(CableReshapeV2, wrappers=[TimeLimit, Monitor], wrappers_args=[
                             {'max_episode_steps': 1000}, {}], render_mode='human',)

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


def posOnlyHarder10(continue_run=False):
    # init_manager(25, 25)
    env_name = CableReshapeHardFlips.__name__

    kwargs = dict(seg_num=10, cable_length=300, scale_factor=800)
    paths = get_paths('cable-reshape-'+str(inspect.stack()[0][3]), 'bigger cable', env_name,
                      continue_run, data=kwargs)

    maker = single_env_maker(CableReshapeHardFlips, wrappers=[TimeLimit, Monitor], wrappers_args=[
                             {'max_episode_steps': 1000}, {}], render_mode='human', **kwargs)

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


# HELPERS


def create_callback_list(paths, save_freq, eval_env) -> tuple:
    checkpoint_callback = CallbackList([SaveModelCallback(
        paths['model_last'], save_freq=save_freq), SaveNormalizeCallback(paths['norm'], save_freq=save_freq)])
    eval_callback = EvalCallback(
        eval_env=eval_env, eval_freq=save_freq, callback_on_new_best=SaveModelCallback(paths['model_best']))
    return checkpoint_callback, eval_callback


if __name__ == "__main__":
    # delete_experiment('cable-reshape-posOnly')
    # posOnly(continue_run=False)
    # forget_last_run('cable-reshape-posOnlyHarder10')
    posOnlyHarder10(continue_run=True)
