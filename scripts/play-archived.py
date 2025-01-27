import json
from argparse import ArgumentParser
from pathlib import Path
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor

import numpy as np
from deform_rl.algos.visualization.players import play_model
from deform_rl.algos.training.training_helpers import single_env_maker


# For playing archived experiment where structure is not complete
from deform_rl.envs.Cable_reshape_env.environment import *
from deform_rl.envs.Rectangle_env.environment import *
from deform_rl.envs.Rectangle_env.debug_env import *
from deform_rl.envs.sim.utils.seed_manager import init_manager


parser = ArgumentParser(prog="play-archived",
                        description="Play a model taht was archived")
parser.add_argument("--norm", type=str, default=None)
parser.add_argument("--env", type=str, default=None)
parser.add_argument('--seed', type=int, default=0)

parser.add_argument("model", type=str)
parser.add_argument("experiment_json", type=str)

args = parser.parse_args()


json_file = json.load(open(args.experiment_json, 'r'))
if args.env is None:
    print("Env not provided, using the one from the json file")
    args.env = json_file['env_name']

if "env_kwargs" in json_file['data']:
    env_kwargs = json_file['data']['env_kwargs']
else:
    env_kwargs = json_file['data']
    print("Using old format for env_kwargs")

normalize = True
if args.norm is None:
    normalize = False

try:
    env_cls = globals()[args.env]
except KeyError:
    print("Env not found")
    exit(1)
maker = single_env_maker(env_cls, seed=args.seed, wrappers=[TimeLimit, Monitor], wrappers_args=[
    {'max_episode_steps': 1000}, {}], render_mode='human', **env_kwargs)

play_model(Path(args.model), Path(args.norm), maker, normalize)
