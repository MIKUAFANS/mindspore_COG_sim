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
PPO training example.
"""

#pylint: disable=C0413
import argparse
import os
from src.ppo import config
from src import PPOTrainer
from mindspore import context
from mindspore import dtype as mstype
from mindspore_rl.core import Session
from mindspore_rl.utils.callback import CheckpointCallback, LossCallback, EvaluateCallback

parser = argparse.ArgumentParser(description='MindSpore Reinforcement PPO')
parser.add_argument('--episode', type=int, default=650, help='total episode numbers.')
parser.add_argument('--device_target', type=str, default='Auto', choices=['Ascend', 'CPU', 'GPU', 'Auto'],
                    help='Choose a device to run the ppo example(Default: Auto).')
parser.add_argument('--precision_mode', type=str, default='fp32', choices=['fp32', 'fp16'],
                    help='Precision mode')
options, _ = parser.parse_known_args()

def train(episode=options.episode):
    '''PPO train entry.'''
    if os.path.exists("my_code/test.txt"):
        os.remove("my_code/test.txt")
    if os.path.exists("my_code/log.log"):
        os.remove("my_code/log.log")
    if options.device_target != 'Auto':
        context.set_context(device_target=options.device_target)
    if context.get_context('device_target') in ['CPU']:
        context.set_context(enable_graph_kernel=True)

    compute_type = mstype.float32 if options.precision_mode == 'fp32' else mstype.float16
    config.algorithm_config['policy_and_network']['params']['compute_type'] = compute_type
    if compute_type == mstype.float16 and options.device_target != 'Ascend':
        raise ValueError("Fp16 mode is supported by Ascend backend.")
    
    # Collect environment information and update replay buffer shape/dtype.
    # So the algorithm could change the environment type without aware of replay buffer schema.
    env_config = config.algorithm_config['collect_environment']
    env = env_config['type'](env_config['params'])
    obs_shape, obs_dtype = env.observation_space.shape, env.observation_space.ms_dtype
    action_shape, action_dtype = env.action_space.shape, env.action_space.ms_dtype
    reward_shape, reward_dtype = env.reward_space.shape, env.reward_space.ms_dtype

    replay_buffer_config = config.algorithm_config['replay_buffer']
    # reward, new_state, action, miu, sigma
    # replay_buffer_config['data_shape'] = [reward_shape, obs_shape, action_shape, obs_shape, (30, 6), (30, 6)]
    replay_buffer_config['data_shape'] = [obs_shape, action_shape, reward_shape, obs_shape, (4, ), (4, )]
    replay_buffer_config['data_type'] = [obs_dtype, action_dtype, reward_dtype, obs_dtype, mstype.float32, mstype.float32]
    # replay_buffer_config['data_type'] = [reward_dtype, obs_dtype, action_dtype, obs_dtype, mstype.float32, mstype.float32]

    context.set_context(mode=context.GRAPH_MODE, max_call_depth=100000)
    ppo_session = Session(config.algorithm_config)
    loss_cb = LossCallback()
    ckpt_cb = CheckpointCallback(50, config.trainer_params['ckpt_path'])
    eval_cb = EvaluateCallback(30)
    cbs = [loss_cb, ckpt_cb, eval_cb]
    ppo_session.run(class_type=PPOTrainer, episode=episode, params=config.trainer_params, callbacks=cbs)

if __name__ == "__main__":
    train()
