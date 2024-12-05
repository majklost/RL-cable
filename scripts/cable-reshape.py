from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


from deform_rl.algos.save_manager import get_paths, consistency_check, delete_experiment
from deform_rl.algos.training.training_helpers import single_env_maker, create_multi_env, SaveNormalizeCallback, SaveModelCallback
from deform_rl.envs.Cable_reshape_env.environment import *

# Multiple cable reshape tasks with slight changes in the environment.
# Each approach is a new function


# HELPERS
def create_callback_list(paths, save_freq, eval_env) -> tuple:
    checkpoint_callback = CallbackList([SaveModelCallback(
        paths['model_last'], save_freq=save_freq), SaveNormalizeCallback(paths['norm'], save_freq=save_freq)])
    eval_callback = EvalCallback(
        eval_env=eval_env, eval_freq=save_freq, callback_on_new_best=SaveModelCallback(paths['model_best']))
    return checkpoint_callback, eval_callback
