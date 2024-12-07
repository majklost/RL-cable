from pathlib import Path
from argparse import ArgumentParser
import numpy as np
from deform_rl.algos.visualization.players import play_model
from deform_rl.algos.save_manager import consistency_check, get_run_paths, load_manager
from deform_rl.algos.training.training_helpers import single_env_maker
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from deform_rl.envs.Cable_reshape_env.environment import *
from deform_rl.envs.sim.utils.seed_manager import init_manager


EXPERIMENTS_PATH = Path(__file__).parent.parent / "experiments"
load_manager(EXPERIMENTS_PATH)


parser = ArgumentParser(prog="play_cable-reshape",
                        description="Play a model on the Cable Reshape environment.")

parser.add_argument("experiment_name", type=str)
parser.add_argument('--seed', type=int, default=-1)
parser.add_argument('--run', type=int, default=-1)
args = parser.parse_args()
consistency_check()
experiment = get_run_paths(args.experiment_name, args.run)
env_name = experiment['env_name']
if args.seed == -1:
    args.seed = np.random.randint(0, 100)
else:
    init_manager(args.seed+10, args.seed+12)

# DEBUG
if not experiment['data']:
    experiment['data'] = dict(seg_num=40, cable_length=300, scale_factor=800)
    # experiment['data'] = dict(seg_num=10, cable_length=300, scale_factor=800)

    # print(f"Playing model for {experiment}")
env_cls = globals()[env_name]
print(f"Playing model for {env_cls.__name__}")
maker = single_env_maker(env_cls, seed=args.seed, wrappers=[TimeLimit, Monitor], wrappers_args=[
    {'max_episode_steps': 1000}, {}], render_mode='human', **experiment['data'])

play_model(experiment['model_best'], experiment['norm'], maker)
