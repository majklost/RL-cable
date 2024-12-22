from pathlib import Path

from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

from torch import nn
from deform_rl.algos.save_manager import get_paths, consistency_check, delete_experiment, load_manager
from deform_rl.algos.training.training_helpers import *
from deform_rl.envs.Rectangle_env.environment import *
from deform_rl.algos.lego.networks import *
# delete_experiment('rect2D')

EXPERIMENTS_PATH = Path(__file__).parent.parent.parent / \
    "experiments"/'debugging'
EXPERIMENTS_PATH.mkdir(exist_ok=True, parents=True)
load_manager(EXPERIMENTS_PATH)

BASE_NAME = 'rect2D-'
