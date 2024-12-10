from argparse import ArgumentParser

from deform_rl.algos.visualization.players import play_model
from deform_rl.algos.training.training_helpers import single_env_maker

from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from deform_rl.envs.Rectangle_env.environment import *

parser = ArgumentParser(prog="play_rect2D",
                        description="Play a model on the Rectangle2D environment.")

parser.add_argument("model_path", type=str)
parser.add_argument("normalize_path", type=str)
parser.add_argument("--seed", type=int, default=20)
args = parser.parse_args()

maker = single_env_maker(RectangleVelDirOnly, seed=args.seed, wrappers=[TimeLimit, Monitor], wrappers_args=[
    {'max_episode_steps': 1000}, {}], render_mode='human')

play_model(args.model_path, args.normalize_path, maker)
