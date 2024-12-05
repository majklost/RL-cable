
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList


from deform_rl.algos.save_manager import get_paths, consistency_check, delete_experiment
from deform_rl.algos.training.training_helpers import single_env_maker, create_multi_env, SaveNormalizeCallback, SaveModelCallback
from deform_rl.envs.Rectangle_env.environment import Rectangle1D
# delete_experiment('rect2D')
consistency_check()
paths = get_paths('rect2Dv3', 'testing', 'Rectangle1D', continue_run=False)
maker = single_env_maker(Rectangle1D, wrappers=[TimeLimit, Monitor], wrappers_args=[
    {'max_episode_steps': 1000}, {}], render_mode='human')
env = create_multi_env(maker, 4, normalize=True)
eval_env = create_multi_env(maker, 1, normalize=True)

SAVE_FREQ = 10000
checkpoint_callback = CallbackList([SaveModelCallback(
    paths['model_last'], save_freq=SAVE_FREQ), SaveNormalizeCallback(paths['norm'], save_freq=SAVE_FREQ)])


eval_callback = EvalCallback(
    eval_env=eval_env, eval_freq=SAVE_FREQ, callback_on_new_best=SaveModelCallback(paths['model_best']))

model = PPO("MlpPolicy", env, verbose=0,
            tensorboard_log=paths['tb'], device='cpu')
print("Training model")
model.learn(total_timesteps=500000, callback=[
            checkpoint_callback, eval_callback])
print("Training done")
