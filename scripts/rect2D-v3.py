from pathlib import Path

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from torch import nn
from deform_rl.algos.save_manager import get_paths, consistency_check, delete_experiment, load_manager
from deform_rl.algos.training.training_helpers import *
from deform_rl.envs.Rectangle_env.environment import *
# delete_experiment('rect2D')

EXPERIMENTS_PATH = Path(__file__).parent.parent / "experiments"
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
load_manager(EXPERIMENTS_PATH)

BASE_NAME = 'rect2D-'


def velObsNoRew():
    env_name = Rectangle1D.__name__
    paths = get_paths(get_name(BASE_NAME), 'first_run',
                      env_name, continue_run=False)
    # maker = single_env_maker(Rectangle1D, wrappers=[TimeLimit, Monitor], wrappers_args=[
    #     {'max_episode_steps': 1000}, {}], render_mode='human')
    # env = create_multi_env(maker, 4, normalize=True)
    # eval_env = create_multi_env(maker, 1, normalize=True)
    env, eval_env = standard_envs(Rectangle1D)
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=800000, callback=[
                ch_clb, ev_clb])
    print("Training done")


def noVel():
    env_name = RectangleNoVel.__name__
    paths = get_paths(get_name(BASE_NAME), 'first_run',
                      env_name, continue_run=False)

    env, eval_env = standard_envs(RectangleNoVel)
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=500000, callback=[
                ch_clb, ev_clb])
    print("Training done")


def velDirOnly():
    env_name = RectangleVelDirOnly.__name__
    paths = get_paths(get_name(BASE_NAME), 'first_run',
                      env_name, continue_run=False)

    env, eval_env = standard_envs(RectangleVelDirOnly)
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=500000, callback=[
                ch_clb, ev_clb])

    print("Training done")


def velRewAfter():
    """
    Learning with velocity in observation first
    and then adding reward shaping
    """
    env_1_name = Rectangle1D.__name__
    env_2_name = RectangleVelReward.__name__
    paths = get_paths(get_name(BASE_NAME), 'first_run',
                      env_2_name, continue_run=False)
    env_1, eval_env_1 = standard_envs(Rectangle1D)
    SAVE_FREQ = 10000
    ch_clb_1, ev_clb_1 = create_callback_list(paths, SAVE_FREQ, eval_env_1)
    model = PPO("MlpPolicy", env_1, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model first part")
    model.learn(total_timesteps=400000, callback=[
                ch_clb_1, ev_clb_1])
    print("Training first part done")

    env_2, eval_env_2 = standard_envs(
        RectangleVelReward, norm_paths=paths['norm'])
    ch_clb_2, ev_clb_2 = create_callback_list(paths, SAVE_FREQ, eval_env_2)
    model.set_env(env_2)
    print("Training model second part")
    model.learn(total_timesteps=400000, callback=[
                ch_clb_2, ev_clb_2], reset_num_timesteps=False)
    print("Training second part done")


def madeForRender():
    env_name = RenderingEnv.__name__
    paths = get_paths(get_name(BASE_NAME), 'first_run',
                      env_name, continue_run=False)
    env, eval_env = standard_envs(RenderingEnv)
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=500000, callback=[
                ch_clb, ev_clb])
    print("Training done")
# Helpers


def madeForRenderTuned():
    env_name = RenderingEnv.__name__
    tuned = {
        'n_steps': 512,
        'batch_size': 32,
        'gamma': 0.99,
        'learning_rate': 0.00016982394779265357,
        'clip_range': 0.3,
        'n_epochs': 10,
        'gae_lambda': 0.95,
        'policy_kwargs': dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256]),
            activation_fn=nn.ReLU,
        ),
    }
    paths = get_paths(get_name(BASE_NAME), 'first_run',
                      env_name, continue_run=False)
    maker = single_env_maker(RenderingEnv, wrappers=[TimeLimit, Monitor], wrappers_args=[
                             {'max_episode_steps': 1000}, {}], render_mode='human')
    env = create_multi_env(maker, 32, normalize=True)
    eval_env = create_multi_env(maker, 1, normalize=True)
    SAVE_FREQ = 20000/32
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0, device='cpu',
                tensorboard_log=paths['tb'], **tuned)
    print("Training model")
    model.learn(total_timesteps=500000, callback=[
                ch_clb, ev_clb])
    print("Training done")


def polar():
    env_name = RectPolar.__name__
    paths = get_paths(get_name(BASE_NAME), 'first_run',
                      env_name, continue_run=False)
    env, eval_env = standard_envs(RectPolar)
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=500000, callback=[
                ch_clb, ev_clb])
    print("Training done")


if __name__ == "__main__":
    # velObsNoRew()
    # noVel()
    # velDirOnly()
    # delete_experiment(BASE_NAME+'velRewAfter')
    # velRewAfter()
    # madeForRender()
    # madeForRenderTuned()
    # delete_experiment(BASE_NAME+'polar')
    polar()
