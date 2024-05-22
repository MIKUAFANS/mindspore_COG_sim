# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
The GymEnvironment base class.
"""

import math
import gym
from gym import spaces
import numpy as np
from mindspore.ops import operations as P
from mindspore_rl.environment.environment import Environment
from mindspore_rl.environment.space import Space

from mindspore_rl.Cogenvdecoder.Cogenvdecoder.CogEnvDecoder import CogEnvDecoder


class CogEnvironment(Environment):
    """
    The GymEnvironment class is a wrapper that encapsulates the Gym(https://gym.openai.com/) to
    provide the ability to interact with Gym environments in MindSpore Graph Mode.

    Args:
        params (dict): A dictionary contains all the parameters which are used in this class.

            +------------------------------+----------------------------+
            |  Configuration Parameters    |  Notices                   |
            +==============================+============================+
            |  name                        |  the name of game in Gym   |
            +------------------------------+----------------------------+
            |  seed                        |  seed used in Gym          |
            +------------------------------+----------------------------+
        env_id (int): A integer which is used to set the seed of this environment.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``

    Examples:
        >>> env_params = {'name': 'CartPole-v0'}
        >>> environment = GymEnvironment(env_params, 0)
        >>> print(environment)
        GymEnvironment<>
    """

    def __init__(self, params, env_id=0):
        super(CogEnvironment, self).__init__()
        self.params = params
        self._name = params['name']
        #self._env = gym.make(self._name)
        self._env = CogEnvDecoder(
            no_graphics=True, 
            # time_scale=1,
            time_scale=5,
            worker_id=env_id,
            seed=1234,
            force_sync=True
        )
        print(self._env.observation_space)
        print(self._env.action_space)
        if 'seed' in params:
            self._env.seed(params['seed'] + env_id * 1000)
        self._observation_space = self._space_adapter(
            self._env.observation_space)
        self._action_space = self._space_adapter(self._env.action_space)
        # self._action_space = self._env.action_space
        self._reward_space = Space((1,), np.float32)
        self._done_space = Space((1,), np.bool_, low=0, high=2)

        # reset op
        reset_input_type = []
        reset_input_shape = []
        reset_output_type = [self._observation_space.ms_dtype,]
        reset_output_shape = [self._observation_space.shape,]
        self._reset_op = P.PyFunc(self._reset, reset_input_type,
                                  reset_input_shape, reset_output_type, reset_output_shape)

        # step op
        step_input_type = (self._action_space.ms_dtype,)
        step_input_shape = (self._action_space.shape,)
        step_output_type = (self.observation_space.ms_dtype,
                            self._reward_space.ms_dtype, self._done_space.ms_dtype)
        step_output_shape = (self._observation_space.shape,
                             self._reward_space.shape, self._done_space.shape)
        self._step_op = P.PyFunc(
            self._step, step_input_type, step_input_shape, step_output_type, step_output_shape)
        self.action_dtype = self._action_space.ms_dtype
        self.cast = P.Cast()
        self.step_num = 0

    def render(self):
        """
        Render the game. Only support on PyNative mode.
        """
        try:
            self._env.render()
        except:
            raise RuntimeError("Failed to render, run in PyNative mode and comment the ms_function.")

    def reset(self):
        """
        Reset the environment to the initial state. It is always used at the beginning of each
        episode. It will return the value of initial state.

        Returns:
            A tensor which states for the initial state of environment.

        """

        return self._reset_op()[0]

    def step(self, action):
        r"""
        Execute the environment step, which means that interact with environment once.

        Args:
            action (Tensor): A tensor that contains the action information.

        Returns:
            - state (Tensor), the environment state after performing the action.
            - reward (Tensor), the reward after performing the action.
            - done (Tensor), whether the simulation finishes or not.
        """

        # Add cast ops for mixed precision case. Redundant cast ops will be eliminated automatically.
        action = self.cast(action, self.action_dtype)
        return self._step_op(action)

    @property
    def observation_space(self):
        """
        Get the state space of the environment.

        Returns:
            The state space of environment.
        """

        return self._observation_space

    @property
    def action_space(self):
        """
        Get the action space of the environment.

        Returns:
            The action space of environment.
        """

        return self._action_space

    @property
    def reward_space(self):
        """
        Get the reward space of the environment.

        Returns:
            The reward space of environment.
        """
        return self._reward_space

    @property
    def done_space(self):
        """
        Get the done space of the environment.

        Returns:
            The done space of environment.
        """
        return self._done_space

    @property
    def config(self):
        """
        Get the config of environment.

        Returns:
            A dictionary which contains environment's info.
        """
        return {}

    def _reset(self):
        """
        The python(can not be interpreted by mindspore interpreter) code of resetting the
        environment. It is the main body of reset function. Due to Pyfunc, we need to
        capsule python code into a function.

        Returns:
            A numpy array which states for the initial state of environment.
        """

        s0, obs = self._env.reset()
        self.step_num = 0
        # In some gym version, the obvervation space is announced to be float32, but get float64 from reset and step.
        s0 = s0.astype(self.observation_space.np_dtype)
        self.state = s0
        self.obs = obs
        self.reward_map = [300] * 5
        self.all_state = [s0.tolist()]
        return s0

    def _step(self, action):
        """
        The python(can not be interpreted by mindspore interpreter) code of interacting with the
        environment. It is the main body of step function. Due to Pyfunc, we need to
        capsule python code into a function.

        Args:
            action(int or float): The action which is calculated by policy net. It could be integer
            or float, according to different environment.

        Returns:
            - s1 (numpy.array), the environment state after performing the action.
            - r1 (numpy.array), the reward after performing the action.
            - done (boolean), whether the simulation finishes or not.
        """
        # next_state, reward, done, obs, flag_ach
        next_state, reward, done, obs, flag_ach = self._env.step(action)
        # print(self.step_num, done, flag_ach)
        
        # 0-17 0, 1, 2| 3, 4, 5| 6, 7, 8| 9, 10, 11| 12, 13, 14| 15, 16, 17|
        if not next_state[5]:
            reward = self.distance(3, 4, self.state, next_state, reward, obs)
        elif not next_state[8]:
            reward = self.distance(6, 7, self.state, next_state, reward, obs)
        elif not next_state[11]:
            reward = self.distance(9, 10, self.state, next_state, reward, obs)
        elif not next_state[14]:
            reward = self.distance(12, 13, self.state, next_state, reward, obs)
        elif not next_state[17]:
            reward = self.distance(15, 16, self.state, next_state, reward, obs)

        # reward += sum(self.reward_map[:int(flag_ach)])
        # self.reward_map[int(flag_ach)] = 0
        self.all_state.append(next_state.tolist())

        # # In some gym version, the obvervation space is announced to be float32, but get float64 from reset and step.
        next_state = next_state.astype(self.observation_space.np_dtype)
        reward = np.array([reward]).astype(np.float32)
        # r = np.array([r]).astype(np.float32)
        # done = np.array([done])

        if flag_ach == 5:
            done = True
            with open('my_code/log.log', 'a') as f:
                f.write(f"Done: {flag_ach}!" + '\n')
            with open('my_code/test.txt', 'a') as f:
                f.write(str(self.all_state) + '\n')
                f.write("==========Done!\n")
        # elif self.step_num == 1000 and flag_ach == 0:
        #    done = True
        #    with open('my_code/log.log', 'a') as f:
        #        f.write(f"Step reached 1000, but did not reach the goal." + '\n')
        # elif self.step_num == 4500:
        #     with open('my_code/log.log', 'a') as f:
        #         f.write(f"Step reached 4500, but did not reach the goal. {flag_ach}" + '\n')
        elif done:
            # if flag_ach >= 4:
            #     with open('my_code/test.txt', 'a') as f:
            #         f.write(str(self.all_state) + '\n')
            #         f.write("==========Done!")
            # if flag_ach > 0:
            #     reward = np.array([-1e6 * abs(np.random.randn())]).astype(np.float32)
            with open('my_code/log.log', 'a') as f:
                f.write(f"Done: {flag_ach}!" + '\n')

        self.state = next_state[:]
        self.obs = obs[:]
        self.step_num += 1
            
        return next_state, reward, done
        

    def _space_adapter(self, gym_space):
        """Transfer gym dtype to the dtype that is suitable for MindSpore"""
        shape = gym_space.shape
        gym_type = gym_space.dtype.type
        # The dtype get from gym.space is np.int64, but step() accept np.int32 actually.
        if gym_type == np.int64:
            dtype = np.int32
        # The float64 is not supported, cast to float32
        elif gym_type == np.float64:
            dtype = np.float32
        else:
            dtype = gym_type

        if isinstance(gym_space, spaces.Discrete):
            return Space(shape, dtype, low=0, high=gym_space.n)

        return Space(shape, dtype, low=gym_space.low, high=gym_space.high)
    
    def distance(self, index_x, index_y, state, next_state, reward, next_obs):
        x_diff = state[index_x] - state[0]
        y_diff = state[index_y] - state[1]
        reward_goal = math.sqrt(x_diff ** 2 + y_diff ** 2)

        x_diff_next = next_state[index_x] - next_state[0]
        y_diff_next = next_state[index_y] - next_state[1]
        reward_next_goal = math.sqrt(x_diff_next ** 2 + y_diff_next ** 2)

        flag1 = (reward_goal > reward_next_goal)  # 下一个状态离目标更近
        flag2 = (abs(reward_goal - reward_next_goal) >= 0.05)  # 这一个动作产生的距离的变化比较大
        # 距离变近并且移动较大奖励比较大
        if flag1 and flag2:
            reward_dist = 50 * (flag1 - 0.5)
        # 距离变近并且移动比较小惩罚较小
        elif flag1 and (not flag2):
            reward_dist = 20 * (flag2 - 0.5)
        # 距离变远并且移动较大惩罚比较大
        elif not flag1 and flag2:
            reward_dist = 60 * (flag1 - 0.5)
        elif not flag1 and not flag2:
            reward_dist = 60 * (flag1 - 0.5)

        return (20 * reward + reward_dist + 10 * ((abs(self.get_theta([state[0], state[1], state[3]], [state[index_x], state[index_y]])) > abs(self.get_theta([next_state[0], next_state[1], next_state[3]], 
            [next_state[index_x], next_state[index_y]]))) - 0.5) - 10 * (next_obs[1][1] > self.obs[1][1]) - 10 * (next_obs[1][0] > self.obs[1][0])) / 1.0
    
    def get_theta(self, self_pos, goal):
        '''
        计算 小车朝向 与 目标 之间的夹角
        '''
        self_pose = self_pos  # [x, y, theta(rad)]
        goal_pose = goal  # [x, y, is_activated?]

        tanTheta = (goal_pose[1] - self_pose[1]) / (goal_pose[0] - self_pose[0])

        # case 1
        if ((goal_pose[1] - self_pose[1] >= 0) and (goal_pose[0] - self_pose[0] >= 0)):
            if ((self_pose[2] <= 0) or ((self_pose[2] >= 0) and (self_pose[2] <= np.pi + np.arctan(tanTheta)))):
                theta = np.arctan(tanTheta) - self_pose[2]
            else:
                theta = np.arctan(tanTheta) - self_pose[2] + 2 * np.pi
        # case 2
        elif ((goal_pose[1] - self_pose[1] >= 0) and (goal_pose[0] - self_pose[0] < 0)):
            if (((self_pose[2] <= 0) and (self_pose[2] >= np.arctan(tanTheta))) or (self_pose[2] >= 0)):
                theta = np.pi + np.arctan(tanTheta) - self_pose[2]
            else:
                theta = np.arctan(tanTheta) - self_pose[2] - np.pi
        # case 3
        elif ((goal_pose[1] - self_pose[1] < 0) and (goal_pose[0] - self_pose[0] < 0)):
            if ((self_pose[2] <= 0) or ((self_pose[2] >= 0) and (self_pose[2] <= np.arctan(tanTheta)))):
                theta = np.arctan(tanTheta) - self_pose[2] - np.pi
            else:
                theta = np.arctan(tanTheta) - self_pose[2] + np.pi
        # case 4
        else:
            if (((self_pose[2] <= 0)) or ((self_pose[2] >= 0) and (self_pose[2] <= np.pi + np.arctan(tanTheta)))):
                theta = np.arctan(tanTheta) - self_pose[2]
            else:
                theta = np.arctan(tanTheta) - self_pose[2] + 2 * np.pi

        return theta
