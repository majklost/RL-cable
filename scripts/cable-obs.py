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
    kwargs = dict(env_kwargs=dict())
    paths = get_paths(get_name(), 'comment', env_name, data=kwargs)

    env, eval_env = standard_envs(CableObsV0, env_kwargs=kwargs['env_kwargs'])
    SAVE_FREQ = 10000
    ch_clb, eval_clb = create_callback_list(paths, SAVE_FREQ, eval_env)
    model = PPO("MlpPolicy", env, verbose=1,
                tensorboard_log=paths['tb'], device='cpu')
    print("Training model")
    model.learn(total_timesteps=5000000, callback=[ch_clb, eval_clb])
    print("Training done")


def get_name():
    return BASE_NAME + str(inspect.stack()[1][3])


if __name__ == "__main__":
    pass
    vanilla()
