# Copyright 2022 Huawei Technologies Co., Ltd
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
DDPG config.
"""

# from mindspore_rl.environment import GymEnvironment
from mindspore_rl.environment.cog_environment import CogEnvironment
from mindspore_rl.core.uniform_replay_buffer import UniformReplayBuffer
from .ddpg import DDPGActor, DDPGLearner, DDPGPolicy

env_params = {'name': 'CartPole-v0'}
eval_env_params = {'name': 'CartPole-v0'}

policy_params = {
    'epsilon': 0.2,
    'state_space_dim': 0,
    'action_space_dim': 0,
    'hidden_size1': 400,
    'hidden_size2': 300,
}

learner_params = {
    'gamma': 0.995,
    'state_space_dim': 0,
    'action_space_dim': 0,
    # 'actor_lr': 1e-4,
    # 'critic_lr': 1e-3,
    'actor_lr': 1e-2,
    'critic_lr': 1e-2,
    'update_factor': 0.05,
    'update_interval': 5
}

trainer_params = {
    'init_collect_size': 1000,
    'duration': 1000,
    'batch_size': 64,
    'ckpt_path': './ckpt',
    'num_eval_episode': 10,
}

actor_params = {
    "damping": 0.15,
    "stddev": 0.2
}

algorithm_config = {
    'actor': {
        'number': 1,
        'type': DDPGActor,
        'params': actor_params,
        'policies': [],
        'networks': ['actor_net'],

    },
    'learner': {
        'number': 1,
        'type': DDPGLearner,
        'params': learner_params,
        'networks': ['actor_net', 'target_actor_net', 'critic_net', 'target_critic_net']
    },
    'policy_and_network': {
        'type': DDPGPolicy,
        'params': policy_params
    },
    'collect_environment': {
        'number': 1,
        # 'type': GymEnvironment,
        'type': CogEnvironment,
        'params': env_params
    },
    'eval_environment': {
        'number': 1,
        # 'type': GymEnvironment,
        'type': CogEnvironment,
        'params': eval_env_params
    },
    'replay_buffer': {
        'number': 1,
        'type': UniformReplayBuffer,
        'capacity': 100000,
        'sample_size': 64
    }
}
