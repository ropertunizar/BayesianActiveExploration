from gym.envs.registration import register

# HalfCheetah (MuJoCo) with two tasks: running and flipping
register(
    id='MagellanHalfCheetah-v2',
    entry_point='envs.half_cheetah:MagellanHalfCheetahEnv',
    max_episode_steps=100
)

# Franka Emika Panda in CoppeliaSim/RLBench with two tasks: push button and move block to target.
# RLBench is only imported when the environment is created.
register(
    id='CoppeliaRLBench-v2',
    entry_point='envs.coppelia:CoppeliaPandaEnv',
    max_episode_steps=251
)

COPPELIA_ENVS = ('CoppeliaRLBench-v2',)
