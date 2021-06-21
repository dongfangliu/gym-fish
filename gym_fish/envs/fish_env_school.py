import gym
from gym import error, spaces, utils
from gym.utils import seeding
import os
import math
from collections import OrderedDict,deque
import numpy as np
from numpy import *
from .py_util import flare_util
from .lib import pyflare as fl
from pathlib import Path
import copy

from .py_util.np_util import generate_traj

from pathlib import *
                
import matplotlib.pyplot as plt      
from mpl_toolkits import mplot3d
import traceback


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


class FishEnvSchool(gym.Env):
    metadata = {}
    def __init__(self, gpuId:int = 0,
                 control_dt=0.2,
                 wa = 0.001,wv = 1.0,
                 couple_mode= fl.COUPLE_MODE.TWO_WAY,
                 action_max = 2,
                 max_time = 10,
                 done_dist=0.1,
                 use_markers=False,
                 rigid_json: str='./py_data/jsons/rigids_2_30.json',
                 fluid_json: str='./py_data/jsons/fluid_param_0.5.json',
                ):
        # store input args
        super().__init__()
        self.gpuId = gpuId
        self.rigid_json = rigid_json
        self.fluid_json = fluid_json
        self.couple_mode = couple_mode
        self.action_max = action_max
        self.max_time = max_time
        self.wv = wv
        self.wa = wa
        self.done_dist = done_dist
        self.control_dt=control_dt
        self.training = True
        self.use_markers= use_markers
        
        
        self.setupDynamcis()
        
        self.seed()
        self.set_datapath('./data/vis_data/')
        
        
        _obs = self.reset()
        self.action_space = self._get_action_space() 
        self.observation_space = convert_observation_to_space(_obs)
        ## Check saving directory avaliable

    def set_datapath(self,folder_path):
        self.simulator.mainDataFolderPath = folder_path
        self.dataPath = {}
        self.dataPath["fluid"] = str(Path(self.simulator.mainDataFolderPath + self.simulator.fluidFolderName + '/').resolve())
        self.dataPath["objects"] = str(Path(self.simulator.mainDataFolderPath + self.simulator.objectsFolderName + '/').resolve())
        self.dataPath["trajectory"] = str(Path(self.simulator.mainDataFolderPath + 'Trajectory/').resolve())
        
        if not os.path.exists(self.simulator.mainDataFolderPath):
            os.makedirs(self.simulator.mainDataFolderPath)
        for p in self.dataPath.values():
            if not os.path.exists(p):
                os.makedirs(p)
                
    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    
    def setupDynamcis(self):
        #fluid
        fluid_param = flare_util.fluid_param()
        fluid_param.from_json(self.fluid_json)
        # Rigid
        rigids = flare_util.rigid_data(gpuId=self.gpuId)
        rigids.from_json(self.rigid_json)
        self.fluid_param = fluid_param
        # generate data from input args
#         self.path_data = path
        self.rigid_data = rigids
        self.simulator = fl.make_simulator(self.fluid_param.data,self.gpuId)
        self.simulator.attachWorld(self.rigid_data.rigidWorld)
        self.simulator.commitInit()
        
        self.dt  = self.rigid_data.rigidWorld.getTimestep()
        self.local_bb_half = np.array([fluid_param.data.width,fluid_param.data.height,fluid_param.data.depth])/2
#         self.simulator.log()
        
    def outOfLocoBB(self):
        rela_dist = np.abs(self.body_xyz -self.target_xyz)
        return (rela_dist>self.local_bb_half).any()
        
        
    def close(self):
        del self.simulator
#         del self.path_data
        del self.rigid_data
        del self.fluid_param
#         del self.renderer
        
        
        
    def stepOnce(self,raw_action,save_fluid=False, save_objects=True):
        
        act = math.sin(6.28*2*self.rigid_data.rigidWorld.time)
        self.free_robot.setCommands(np.ones(self.free_robot_action_dim)*act)
        
        self.chase_robot.setCommands(raw_action)
        self.simulator.step(self.couple_mode)
        obs = self._get_obs()        
        
        self.free_robot_trajectory_points.append(self.free_robot.getBaseLink().getPosition())
        self.chase_robot_trajectory_points.append(self.chase_robot.getBaseLink().getPosition())

        done = self._get_done() 
        if (not np.isfinite(obs).all()):
            print("NAN OBS")
            obs = self.last_obs
            done=True
        self.save_data_framRate(save_fluid,save_objects)
        self.last_obs = obs
        return obs,None,done,None
    
    def step(self, action,save_fluid=False, save_objects=False):
        dt = self.rigid_data.rigidWorld.getTimestep()
        t = 0
        self.last_action = action
        act= self._unnormalize_action(action)
        while t< self.control_dt:
            obs,_,done,_ =self.stepOnce(act,save_fluid,save_objects)
            t = t+dt
            if done:
                if not self.training:
                    continue;
                else:
                    break
        reward,info = self._get_reward(self.control_dt,t,obs = obs,action=action)
        return obs, reward, done,info

    def stepSave(self,action,save_fluid=False, save_objects=True):
        o,r,d,i = self.step(action,save_fluid, save_objects)
        return o,r,d,i
    

    def _get_done(self):
        done = False 
        done = done or self.rigid_data.rigidWorld.time>self.max_time 
        done = done or self.outOfLocoBB()
        done = done or self.walk_target_dist<self.done_dist 
        done = done or self.walk_target_dist>1.2
        return  done 

    
    def _get_reward(self,t_standard,t, obs=None, action=None):
#         dist_potential_old = self.dist_potential
#         self.dist_potential = self.calc__dist_potential()

#         vel_reward = self.wv*np.exp(-5* np.abs(np.linalg.norm(self.vel)-self.target_vel))
        
        total_reward = self.wv*np.exp(-np.linalg.norm(self.dvel)/0.2)-self.wa*self.force_use
            
        info = {'energy_reward':self.wa*self.force_use,"vel_reward":self.wv*np.exp(-np.linalg.norm(self.dvel)/0.2)}
        return total_reward,info
    
    def _get_action_space(self):
        low = -np.ones(self.action_dim)*self.action_max
        high= -low
        self.action_space_mean = (low+high)/2
        self.action_space_std = (high-low)/2
        return spaces.Box(low = -1,high=1,shape=low.shape)
    
    def _normalize_action(self,action):
        return np.clip((action-self.action_space_mean)/self.action_space_std,-1,1)
    def _unnormalize_action(self,action):
        action = np.clip(action,-1,1)
        return action*self.action_space_std+self.action_space_mean
    
        
    def _get_obs(self):
        # obs
        self.body_xyz =  self.chase_robot.getCOM()
        self.target_xyz = self.free_robot.getCOM()
        
        self.dvel  =  self.chase_robot.getCOMLinearVelocity()-self.free_robot.getCOMLinearVelocity()
        
        
        # update local matrix
        x_axis = self.chase_robot.getBaseLinkFwd()
        y_axis = self.chase_robot.getBaseLinkUp()
        z_axis = self.chase_robot.getBaseLinkRight()
        self.world_to_local = np.linalg.inv(np.array([x_axis,y_axis,z_axis]).transpose())
        
                             
        self.walk_target_dist = np.linalg.norm(self.body_xyz-self.target_xyz)
        
        #in local coordinate
        dp_local = np.dot(self.world_to_local,np.transpose(self.target_xyz-self.body_xyz))
        dvel_local = np.dot(self.world_to_local,np.transpose(self.dvel))
        
        joint_pos = self.chase_robot.getPositions(includeBase=False)
        joint_vel = self.chase_robot.getVelocities(includeBase=False)
        
        self.force_use= np.mean(np.abs(self.last_action))
        
        
        if self.use_markers:
            
            markers = self.chase_robot.getMarkers()
            
            markers_pressure = np.array(markers.getMarkersPressure())
            
            markers_vel = np.array(markers.getMarkersVel()).flatten()
            
            obs = np.concatenate(
                (
                    dvel_local,
                    self.last_action,
                    np.array([self.force_use]),
                    markers_vel,
                    markers_pressure,
                    joint_pos/0.52,
                    joint_vel/10,

            ),axis=0)
            
        else:
            obs = np.concatenate(
                (
                    dvel_local,
                    self.last_action,
                    np.array([self.force_use]),

                    joint_pos/0.52,
                    joint_vel/10,

            ),axis=0)
        return obs
    

        
        
    def _reset_robot(self):
        self.setupDynamcis()
        self.free_robot = self.rigid_data.skeletons[0].dynamics
        self.chase_robot = self.rigid_data.skeletons[1].dynamics
        
        self.chase_robot.setHead(np.array([-0.5,0,0]),np.array([1,0,0]))
        
        self.action_dim = self.chase_robot.getNumDofs()
        self.free_robot_action_dim = self.chase_robot.getNumDofs()
#         angle =  self.np_random.uniform(self.vel_theta[0],self.vel_theta[1])
#         vel =  np.array([math.cos(angle),0,math.sin(angle)])*self.np_random.uniform(self.random_vel[0],self.random_vel[1],size=1)
        
#         skeleton_dynamics.getJoint("head").setVelocity(0,vel[0])
#         skeleton_dynamics.getJoint("head").setVelocity(1,vel[1])
# #         skeleton_dynamics.getJoint("head").setPosition(2,self.np_random.uniform(-0.52,0.52,size=1))
#         joint_list =['spine','spine01','spine02','spine03']
#         for jnt_name in joint_list:
#             skeleton_dynamics.getJoint(jnt_name).setPosition(0,self.np_random.uniform(-0.52,0.52,size=1))
#             skeleton_dynamics.getJoint(jnt_name).setVelocity(0,self.np_random.uniform(-10,10,size=1))
#         skeleton_dynamics.update()
        frame = self.free_robot.getBaseLink().getFrame()
        self.chase_robot.setRefFrame(frame)
        
        self.body_xyz = self.chase_robot.getBaseLink().getPosition()
        self.init_pos = self.body_xyz
    def set_task(self,goal_pos,target_vel):
        x_axis = self.chase_robot.getBaseLinkFwd()
        y_axis = self.chase_robot.getBaseLinkUp()
        z_axis = self.chase_robot.getBaseLinkRight()
        
        self.world_to_local = np.linalg.inv(np.array([x_axis,y_axis,z_axis]).transpose())
        self.dp_init = np.dot(self.world_to_local,np.transpose(self.free_robot.getBaseLink().getPosition()-self.init_pos))
        
    def _reset_task(self):
        pass
#         theta = self.np_random.uniform(self.theta[0],self.theta[1])

#         goal_dir = np.array([math.cos(theta),0,math.sin(theta)])
#         goal_pos = self.init_pos+self.radius*goal_dir
    
#         target_vel = self.np_random.uniform(self.random_vel[0],self.random_vel[1])
        
#         self.set_task(goal_pos,target_vel)
        
        
    def reset(self):

        
        self._reset_robot()
        self._reset_task()
        
        self.free_robot_trajectory_points=[]
        self.chase_robot_trajectory_points=[]
        
        self.last_action = np.ones(self.action_dim)*0
        self.last_obs = self._get_obs()
        
        return self.last_obs
    
    
    
    def plot3d(self,title=None,fig_name=None,elev=45,azim=45):

        path_points = np.array(self.free_robot_trajectory_points+self.chase_robot_trajectory_points)
        ax=plt.figure().add_subplot(111, projection = '3d')
        X = path_points[:,0]
        Y = path_points[:,1]
        Z = path_points[:,2]
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max() / 2.0
        mid_x = (X.max()+X.min()) * 0.5
        mid_y = (Y.max()+Y.min()) * 0.5
        mid_z = (Z.max()+Z.min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_z - max_range, mid_z + max_range)
        ax.set_zlim(mid_y - max_range, mid_y + max_range)
        
        if self.free_robot_trajectory_points!=None:
            ax.scatter3D(xs=[x[0] for x in self.free_robot_trajectory_points],
                zs=[x[1] for x in self.free_robot_trajectory_points],
                ys=[x[2] for x in self.free_robot_trajectory_points],
                c=[[0,i/len(self.free_robot_trajectory_points),0] for i in range(len(self.free_robot_trajectory_points))])
        if self.chase_robot_trajectory_points!=None:
            ax.scatter3D(xs=[x[0] for x in self.chase_robot_trajectory_points],
                zs=[x[1] for x in self.chase_robot_trajectory_points],
                ys=[x[2] for x in self.chase_robot_trajectory_points],
                c=[[0,0,i/len(self.chase_robot_trajectory_points)] for i in range(len(self.chase_robot_trajectory_points))])
        ax.view_init(elev=elev,azim=azim)#改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        if title!=None:
            ax.set_title(title)
        if fig_name!=None:
            plt.savefig(fig_name)
        plt.show()
    def render(self, mode='human'):
        if self.renderer!=None:
            self.renderer.render()
    def save_data(self, save_fluid=False, save_objects=True,frame_num = 0):
#         ref_line = fl.debugLine()
#         ref_line.vertices = self.trajectory_points
#         fl.VTKWriter.writeLines([ref_line], self.dataPath["trajectory"]+"/trajectory_actual%05d.vtk"% frame_num)
        if save_fluid:
            fluid_name = "fluid%05d" % frame_num
            self.simulator.saveFluidData(fluid_name)
        if save_objects:
            objects_name = "object%05d" % frame_num
            self.simulator.saveObjectsData(objects_name)
        
    def save_data_framRate(self, save_fluid=False, save_objects=True,framRate = 30):
        if self.simulator.getIterNum()%self.simulator.getIterPerSave(framRate)!=0:
            return
        frame_num=(self.simulator.getIterNum() / self.simulator.getIterPerSave(framRate))
        self.save_data(save_fluid, save_objects,frame_num)
