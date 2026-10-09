"""
Franka Emika Panda in CoppeliaSim (RLBench `slide_block_to_target` scene), controlled with joint velocities and a
discrete gripper action (|A| = 8). The low-dimensional RLBench observation is used as state (|S| = 43). Indices
used by the rewards:

    state[22:25]   gripper (end-effector) position
    state[-6:-3]   block position
    state[-3:]     target position

Two tasks are defined on the same scene, so that one exploration buffer serves both:

    push_button  -- press the green target pad with the gripper ("PushButton" in the paper)
    move_block   -- push the red block onto the green target ("MoveBlockToTarget" in the paper)

Both rewards are 100 on success and a smooth, bounded proximity shaping otherwise.
"""
import torch
from rlbench.gym.rlbench_env import RLBenchEnv
from rlbench.utils import name_to_task_class

from utilities import Measure
from .task import Task, RewardFunction

SUCCESS_THRESHOLD = 0.01          # distance (m) to the target that counts as success
SUCCESS_REWARD = 100

EE, BLOCK, TARGET = slice(22, 25), slice(-6, -3), slice(-3, None)


def _distance(x, y):
    return (x - y).pow(2).sum(dim=-1).sqrt()


"""
Push button
"""


def push_button_reward(next_states):
    distances = _distance(next_states[..., EE], next_states[..., TARGET])
    rewards = 50 * torch.sigmoid(-10 * distances)
    return torch.where(distances < SUCCESS_THRESHOLD, SUCCESS_REWARD * torch.ones_like(rewards), rewards)


class PushButtonMeasure(Measure):
    def __call__(self, states, actions, next_states, next_state_means, next_state_vars, model):
        return push_button_reward(next_states)


class PushButtonRewardFunction(RewardFunction):
    def __call__(self, state, action, next_state):
        return push_button_reward(torch.tensor(next_state)).item()


"""
Move block to target
"""


def move_block_reward(next_states):
    ee = next_states[..., EE]
    block = next_states[..., BLOCK]
    target = next_states[..., TARGET]

    distances_block2target = _distance(block, target)

    # pushing pose: 5 cm behind the block, on the opposite side of the target, at table height
    push_pose = block + (block - target) * 0.05 / distances_block2target.unsqueeze(-1)
    push_pose[..., 2] = target[..., 2]
    distances_ee2push = _distance(ee, push_pose)

    # approach the pushing pose, and only then get rewarded for bringing the block to the target
    rewards_ee2block = 25 * torch.sigmoid(-10 * distances_ee2push)
    rewards_block2target = 50 * torch.sigmoid(-10 * distances_block2target)
    rewards_block2target = torch.where(distances_ee2push < 0.07, rewards_block2target,
                                       torch.zeros_like(rewards_block2target))
    rewards = rewards_ee2block + rewards_block2target

    return torch.where(distances_block2target < SUCCESS_THRESHOLD, SUCCESS_REWARD * torch.ones_like(rewards), rewards)


class MoveBlockMeasure(Measure):
    def __call__(self, states, actions, next_states, next_state_means, next_state_vars, model):
        return move_block_reward(next_states.clone().detach())


class MoveBlockRewardFunction(RewardFunction):
    def __call__(self, state, action, next_state):
        return move_block_reward(torch.tensor(next_state)).item()


"""
Environment
"""


class CoppeliaPandaEnv(RLBenchEnv):
    def __init__(self, observation_mode='state', render_mode=None):
        super().__init__(name_to_task_class('slide_block_to_target'), observation_mode=observation_mode,
                         render_mode=render_mode)

    @property
    def tasks(self):
        return {'push_button': Task(measure=PushButtonMeasure(), reward_function=PushButtonRewardFunction()),
                'move_block': Task(measure=MoveBlockMeasure(), reward_function=MoveBlockRewardFunction())}
