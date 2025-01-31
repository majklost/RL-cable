from deform_rl.algos.training.training_helpers import *
from deform_rl.envs.Cable_obs_env.environment import *
from deform_rl.envs.sim.utils.environment_controller import EnvironmentController


def test_movement():
    maker = single_env_maker(EmptyNoRewShorter, wrappers=[TimeLimit, Monitor], wrappers_args=[
        {'max_episode_steps': 1000}, {}], render_mode='human')
    env = create_multi_env(maker, 1, normalize=True)
    assert len(env.action_space.shape) == 1, "Only 1D action space is supported"
    assert env.action_space.shape[0] % 2 == 0, "Only even number of actions is supported"
    controller = EnvironmentController(env.action_space.shape[0] // 2)
    obs = env.reset()
    while True:
        action = np.expand_dims(controller.get_action(), 0)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            print("Episode done")
        if pygame.event.get(pygame.QUIT):
            break


if __name__ == "__main__":
    test_movement()
