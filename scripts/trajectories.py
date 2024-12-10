import matplotlib.pyplot as plt
from pathlib import Path
from argparse import ArgumentParser
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
# import random


from deform_rl.envs.Rectangle_env.environment import *
from deform_rl.algos.training.training_helpers import *
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from deform_rl.algos.save_manager import consistency_check, get_run_paths, load_manager

parser = ArgumentParser(prog="play_cable-reshape",
                        description="Play a model on the Cable Reshape environment.")

parser.add_argument("experiment_name", type=str)
parser.add_argument('--num', type=int, default=40)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()
EXPERIMENTS_PATH = Path(__file__).parent.parent / "experiments"
load_manager(EXPERIMENTS_PATH)

# Use trained rect2D model to visualize which paths it takes
# Assuming training with relative paths
experiment = get_run_paths(args.experiment_name, -1)

ENV_CLS = RenderingEnv

maker = single_env_maker(ENV_CLS, seed=args.seed, wrappers=[TimeLimit, Monitor], wrappers_args=[
    {'max_episode_steps': 1000}, {}], render_mode='human')
model = PPO.load(experiment['model_best'], device='cpu')
env = create_multi_env(maker, 1, normalize_path=experiment['norm'])
if experiment['norm'] is not None:
    env.training = False
else:
    print("No VecNormalize statistics found. Playing without normalization.")

trajectories = []
for i in range(args.num):
    obs = env.reset()
    trajectory = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        trajectory.append(info[0]['position'])
        # env.render()
        if done:
            break
    trajectories.append(trajectory)
env.close()
print(len(trajectories))
print(len(trajectories[0]))

for trajectory in trajectories:
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1])

plt.xlabel('X position')
plt.ylabel('Y position')
plt.title('Trajectories')
for trajectory in trajectories:
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], linewidth=2, marker='None')

plt.show()
