from gym.envs.registration import register

register(
    id='fish-basic-v0',
    entry_point='gym_fish.envs:FishEnvBasic',
)
register(
    id='fish-collision-avoidance-v0',
    entry_point='gym_fish.envs:FishEnvCollisionAvoidance',
)
register(
    id='fish-pose-control-v0',
    entry_point='gym_fish.envs:FishEnvPoseControl',
)
register(
    id='fish-schooling-v0',
    entry_point='gym_fish.envs:FishEnvSchooling',
)
register(
    id='fish-vel-v0',
    entry_point='gym_fish.envs:FishEnvVel',
)

