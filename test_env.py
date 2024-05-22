
from mlagents_envs.environment import UnityEnvironment
from gym_unity.envs import UnityToGymWrapper
env_name = "/root/ruijun.zhang/homework2-2-baseline/reinforcement/example/dqn/mindspore_rl/Cogenvdecoder/Cogenvdecoder/linux_v3.1/cog_sim2real_env.x86_64"
unity_env = UnityEnvironment(env_name)

self._env = UnityToGymWrapper(unity_env, allow_multiple_obs=True, uint8_visual=True)
