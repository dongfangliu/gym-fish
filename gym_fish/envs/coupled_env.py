from typing import Any, Dict, OrderedDict, Tuple
import abc
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym_fish.envs.entities.coupled_sim import coupled_sim
from gym_fish.envs.entities.fluid_solver import fluid_solver
from gym_fish.envs.entities.rigid_solver import rigid_solver
import numpy as np
import gym_fish.envs.py_util.flare_util as fl_util
import gym_fish.envs.lib.pyflare as fl
def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(OrderedDict([
            (key, convert_observation_to_space(value))
            for key, value in observation.items()
        ]))
    elif isinstance(observation, np.ndarray):
        low = np.full(observation.shape, -float('inf'))
        high = np.full(observation.shape, float('inf'))
        space = spaces.Box(low, high, dtype=observation.dtype)
    else:
        raise NotImplementedError(type(observation), observation)

    return space
class coupled_env(gym.Env):
    def __init__(self,fluid_json:str,rigid_json:str,gpuId:int,couple_mode:fl.COUPLE_MODE =fl.COUPLE_MODE.TWO_WAY) -> None:
        super().__init__()
        # here init dynamics ,action_space and observation space
        self.fluid_json =fluid_json
        self.rigid_json = rigid_json
        self.gpuId =  gpuId
        self.couple_mode  = couple_mode
        self.resetDynamics()
        self.seed()
        _obs = self.reset()
        self.action_space = self._get_action_space()
        self.observation_space = convert_observation_to_space(_obs)

    def resetDynamics(self):
        fluid_param =fl_util.fluid_param()
        fluid_param.from_json(self.fluid_json)
        rigids_data = fl_util.rigid_data(gpuId=self.gpuId)
        rigids_data.from_json(self.rigid_json)
        rigid_solv = rigid_solver(rigids_data)
        fluid_solv = fluid_solver(fluid_param = fluid_param,gpuId=self.gpuId,couple_mode=self.couple_mode)
        self.simulator = coupled_sim(fluid_solv,rigid_solv)
    
    def _get_action_space(self):
        low = self.simulator.rigid_solver.get_action_lower_limits()
        high= self.simulator.rigid_solver.get_action_upper_limits()
        return spaces.Box(low = low,high=high,shape=low.shape)
    def close(self) -> None:
        del self.simulator
    @abc.abstractmethod
    def _update_state(self)->None:
        pass
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
    def step(self, action) :
        self._step(action)
        obs = self._get_obs()
        done= self._get_done()
        reward,info = self._get_reward(obs,action)
        return obs,reward,done,info
    @abc.abstractmethod
    def _step(self, action)->None:
        pass
    @abc.abstractmethod
    def _get_obs(self)->np.array:
        self._update_state()
        pass
    @abc.abstractmethod
    def _get_reward(self,cur_obs,cur_action):
        pass
    @abc.abstractmethod
    def _get_done(self)->bool:
        pass
    @abc.abstractmethod
    def reset(self) ->np.array:
        pass




