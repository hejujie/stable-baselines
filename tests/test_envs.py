import pytest
import gym

from stable_baselines.common.env_checker import check_env
from stable_baselines.common.bit_flipping_env import BitFlippingEnv


@pytest.mark.parametrize("env_id", ['CartPole-v0', 'Pendulum-v0', 'BreakoutNoFrameskip-v4'])
def test_env(env_id):
    """
    Check that environmnent integrated in Gym pass the test.

    :param env_id: (str)
    """
    env = gym.make(env_id)
    with pytest.warns(None) as record:
        check_env(env)

    # Pendulum-v0 will produce a warning because the action space is
    # in [-2, 2] and not [-1, 1]
    if env_id == 'Pendulum-v0':
        assert len(record) == 1


def test_custom_envs():
    goal_env = BitFlippingEnv()
    check_env(goal_env)

    # Check failures
