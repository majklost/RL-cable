import optuna

from pathlib import Path
from stable_baselines3 import PPO
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from deform_rl.algos.training.training_helpers import single_env_maker, create_multi_env, SaveNormalizeCallback, SaveModelCallback
from deform_rl.envs.Cable_reshape_env.environment import *
from deform_rl.envs.sim.utils.seed_manager import init_manager
from deform_rl.algos.training.hyperparams import give_args

PATH = Path(__file__).parent.parent / "experiments" / "hyperopt"
TIMESTEPS = 70000


class PruneCallback(BaseCallback):
    """
    Stop the training once a threshold in episodic reward
    has been reached (i.e. when the model is good enough).

    It must be used with the ``EvalCallback``.

    :param reward_threshold:  Minimum expected reward per episode
        to stop training.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because episodic reward
        threshold reached
    """

    parent: EvalCallback

    def __init__(self, verbose: int = 0, trial: optuna.Trial = None):
        super().__init__(verbose=verbose)
        self.trial = trial

    def _on_step(self) -> bool:
        assert self.parent is not None, "``StopTrainingOnMinimumReward`` callback must be used with an ``EvalCallback``"
        intermediate_value = self.parent.best_mean_reward
        self.trial.report(intermediate_value, step=self.num_timesteps)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
        return True


def objectivePosOnlyReshape(trial):
    env_name = CableReshapeV2.__name__
    kwargs = dict(seg_num=10, cable_length=300, scale_factor=800)
    maker = single_env_maker(CableReshapeV2, wrappers=[TimeLimit, Monitor], wrappers_args=[
                             {'max_episode_steps': 1000}, {}], render_mode='human', **kwargs)

    n_envs = trial.suggest_categorical("n_envs", [4, 16, 32])
    env = create_multi_env(maker, n_envs, normalize=True)
    eval_env = create_multi_env(maker, 1, normalize=True)
    save_norm_clb = SaveNormalizeCallback(PATH / "norm", save_freq=1)
    prune_clb = PruneCallback(trial=trial)
    eval_clb = EvalCallback(eval_env, best_model_save_path=PATH /
                            "best_model", callback_on_new_best=save_norm_clb, callback_after_eval=prune_clb, n_eval_episodes=10)

    args = give_args(trial)
    model = PPO("MlpPolicy", env, verbose=0, device='cpu', **args)
    model.learn(total_timesteps=TIMESTEPS, callback=[save_norm_clb, eval_clb])
    rew, std = evaluate_policy(model, eval_env, n_eval_episodes=10)
    return rew


if __name__ == "__main__":
    study = optuna.load_study(
        study_name="PosOnlyReshape", storage=f"sqlite:///{PATH / 'PosOnlyReshape.db'}")
    study.optimize(objectivePosOnlyReshape, n_trials=2,
                   timeout=23*3600, show_progress_bar=True)
