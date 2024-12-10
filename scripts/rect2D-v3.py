from pathlib import Path

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


from deform_rl.algos.save_manager import get_paths, consistency_check, delete_experiment, load_manager
from deform_rl.algos.training.training_helpers import single_env_maker, create_multi_env, SaveNormalizeCallback, SaveModelCallback, get_name, create_callback_list
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
    maker = single_env_maker(Rectangle1D, wrappers=[TimeLimit, Monitor], wrappers_args=[
        {'max_episode_steps': 1000}, {}], render_mode='human')
    env = create_multi_env(maker, 4, normalize=True)
    eval_env = create_multi_env(maker, 1, normalize=True)
    SAVE_FREQ = 10000
    ch_clb, ev_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=0,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=500000, callback=[
                ch_clb, ev_clb])
    print("Training done")


def noVel():
    env_name = RectangleNoVel.__name__
    paths = get_paths(get_name(BASE_NAME), 'first_run',
                      env_name, continue_run=False)
    maker = single_env_maker(RectangleNoVel, wrappers=[TimeLimit, Monitor], wrappers_args=[
        {'max_episode_steps': 1000}, {}], render_mode='human')
    env = create_multi_env(maker, 4, normalize=True)
    eval_env = create_multi_env(maker, 1, normalize=True)
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
    maker = single_env_maker(RectangleVelDirOnly, wrappers=[TimeLimit, Monitor], wrappers_args=[
        {'max_episode_steps': 1000}, {}], render_mode='human')
    env = create_multi_env(maker, 4, normalize=True)
    eval_env = create_multi_env(maker, 1, normalize=True)
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
    velDirOnly()
