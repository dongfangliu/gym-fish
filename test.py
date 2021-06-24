
import matplotlib.pyplot as plt
import os
import time
import gym
import numpy as np
import math
from pathlib import Path

import gym_fish
from gym_fish.envs.lib import pyflare as fl

gpuId = 0

control_dt=0.2
theta = np.array([0,0])
radius=1.0
max_time = 10
action_max= 10
done_dist = 0.15
dist_distri_param =np.array([0,0.0])

# couple_mode = fl.COUPLE_MODE.EMPIRICAL
couple_mode =  fl.COUPLE_MODE.TWO_WAY

ratio = 0.005/control_dt*(max_time/10)
use_com=False

wc = 0.0*np.array([1.0,0.5])
wp = 1.0*np.array([0.0,1.0])
wa = 0.5

env=gym.make('fish-basic-v0',
                  gpuId=gpuId,
                        couple_mode=couple_mode,
             control_dt=control_dt,
                        radius=radius,
                       theta=theta,action_max=action_max,
                        max_time=max_time,
                        wp=wp,wc=wc,wa=wa,
                       done_dist=done_dist,dist_distri_param=dist_distri_param
                   )
action = env.action_space.sample()
# env.step(action)
# pass
# env2= gym.make('fish-vel-v0')
# action = env2.action_space.sample()
# env2.step(action)
arr = env.render(mode='human')
env.simulator.save(True,True)

