from sys import path
from typing import Any, Dict, Tuple
from .coupled_env import coupled_env
from .lib import pyflare as fl
from .py_util import np_util as np_util 
import numpy as np
import os
import math
import json


from matplotlib.patches import FancyArrowPatch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.proj3d import proj_transform
from mpl_toolkits.mplot3d.axes3d import Axes3D
class Arrow3D(FancyArrowPatch):
    def __init__(self, x, y, z, dx, dy, dz, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._xyz = (x,y,z)
        self._dxdydz = (dx,dy,dz)

    def draw(self, renderer):
        x1,y1,z1 = self._xyz
        dx,dy,dz = self._dxdydz
        x2,y2,z2 = (x1+dx,y1+dy,z1+dz)

        xs, ys, zs = proj_transform((x1,x2),(y1,y2),(z1,z2), renderer.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        super().draw(renderer)
def _arrow3D(ax, x, y, z, dx, dy, dz, *args, **kwargs):
    '''Add an 3d arrow to an `Axes3D` instance.'''
    arrow = Arrow3D(x, y, z, dx, dy, dz, *args, **kwargs)
    ax.add_artist(arrow)
setattr(Axes3D,'arrow3D',_arrow3D)

class FishEnvPoseControl(coupled_env):
    def __init__(self, 
                control_dt=0.2,
                wr=1,
                wa = 0.5,
                max_time = 10,
                # theta should be in the range of [0,180]
                theta = np.array([90,90]),
                # phi should be in the range of [0,360]
                phi = np.array([0,0]),
                data_folder = "./data/vis_data/",
                env_json :str = '../assets/env_file/env_pose_control.json',
                gpuId: int=0,
                couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY) -> None:
        self.wr = wr
        self.wa = wa
        self.theta = theta/180.0*math.pi
        self.phi = phi/180.0*math.pi
        self.control_dt=control_dt
        self.max_time = max_time
        self.save=False
        # use parent's init function to init default env data, like action space and observation space, also init dynamics
        super().__init__(data_folder,env_json, gpuId, couple_mode=couple_mode)


    def _step(self, action) -> None:
        t = 0
        agent = self.simulator.rigid_solver.get_agent(0)
        self.trajectory_poses.append((agent.com,agent.fwd_axis,agent.up_axis,agent.right_axis))
        while t<self.control_dt:
            self.simulator.iter(action)
            t = t+self.simulator.dt
            if self.save:
                self.save_at_framerate(True,False)
            if not np.isfinite(self._get_obs()).all():
                break
    def _get_reward(self, cur_obs, cur_action) :
        ori_reward = - np.linalg.norm( self.r_diff_vec,ord=1)*self.wr
        
        cur_action =self.normalize_action(cur_action)
        action_reward = -np.sum(np.abs(cur_action)**0.5)*self.wa
        
        total_reward = ori_reward+action_reward
        
        info = {'ori_reward':ori_reward,"action_reward":action_reward}
        return total_reward,info

    def _get_done(self) -> bool:
        done = False 
        done = done or self.simulator.time>self.max_time 
        done = done or (not np.isfinite(self._get_obs()).all())
        return  done 
    def normalize_action(self,action):
        action_space_mean = (self.action_space.low+self.action_space.high)/2
        action_space_std = (self.action_space.high-self.action_space.low)/2
        return np.clip((action-action_space_mean)/action_space_std,-1,1)
    def _get_obs(self) -> np.array:
        self._update_state()
        agent = self.simulator.rigid_solver.get_agent(0)
#         proj_pt_local = np.dot(self.world_to_local,np.transpose(self.proj_pt_world-self.body_xyz))
        if agent.has_buoyancy:
            scalar_obs  = np.array([agent.bcu.bladder_volume])
        else:
            scalar_obs= np.array([])
        obs = np.concatenate(
            (
                scalar_obs,
                self.r_diff_vec,
                agent.positions/0.52,
                agent.velocities/10,
        ),axis=0)
        if np.isfinite(obs).all():
            self.last_obs = obs
        return self.last_obs

    def _update_state(self):
        agent = self.simulator.rigid_solver.get_agent(0)
        self.body_xyz =  agent.com
        x_axis = agent.fwd_axis
        y_axis = agent.up_axis
        z_axis = agent.right_axis
        rot_mat =np.array([x_axis,y_axis,z_axis]).transpose()
        R_diff = np.dot(self.r_desired,rot_mat.transpose())
        self.r_diff_vec = 0.5*np.array([R_diff[2,1]-R_diff[1,2],R_diff[0,2]-R_diff[2,0],R_diff[1,0]-R_diff[0,1]])
        

    def set_task(self,theta,phi):
        agent = self.simulator.rigid_solver.get_agent(0)
        self.init_pos = agent.com
        fwd_vec = np.array([math.sin(theta)*math.cos(phi),math.sin(theta)*math.sin(phi),math.cos(theta)])
        up_vec = np.array([0,1,0])
        right_vec = np.cross(fwd_vec,up_vec)
        up_vec = np.cross(right_vec,fwd_vec)
        self.r_desired = np.array([fwd_vec,up_vec,right_vec]).transpose()
        
        
        
    def _reset_task(self):
        agent = self.simulator.rigid_solver.get_agent(0)
        agent.bcu.reset(randomize=False)
        theta = self.np_random.uniform(self.theta[0],self.theta[1])
        phi = self.np_random.uniform(self.phi[0],self.phi[1])
       
        self.set_task(theta,phi)


    def reset(self) -> Any:
        self.resetDynamics()
        self._reset_task()
        self.trajectory_poses=[]
        self._update_state()
        self.last_obs = self._get_obs()
        return self._get_obs()

    def plot3d(self, title=None, fig_name=None, elev=45, azim=45):
        import matplotlib.pyplot as plt  
        ax = plt.figure().add_subplot(111, projection='3d')
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2)
        if self.trajectory_poses != None:
            trajectory_poses =self.trajectory_poses
            for i in range(0,len(trajectory_poses)):
                body_xyz = trajectory_poses[i][0]
                fwd_axis = trajectory_poses[i][1]
                up_axis = trajectory_poses[i][2]
                right_axis = trajectory_poses[i][3]
#                 ax.scatter3D(xs = body_xyz[0],ys=body_xyz[1],zs=body_xyz[2],c=(0, 0, i / len(trajectory_poses)))
                ax.arrow3D(body_xyz[0],body_xyz[2],body_xyz[1],fwd_axis[0],fwd_axis[2],fwd_axis[1],arrowstyle="-|>",mutation_scale=20,ec =(0,0,0,i/len(trajectory_poses)),fc=(1,0,0,i/len(trajectory_poses)))
                ax.arrow3D(body_xyz[0],body_xyz[2],body_xyz[1],right_axis[0],right_axis[2],right_axis[1],arrowstyle="-|>",mutation_scale=20,ec =(0,0,0,i/len(trajectory_poses)),fc=(0,1,0,i/len(trajectory_poses)))
                ax.arrow3D(body_xyz[0],body_xyz[2],body_xyz[1],up_axis[0],up_axis[2],up_axis[1],arrowstyle="-|>",mutation_scale=20,ec =(0,0,0,i/len(trajectory_poses)),fc=(0,0,1,i/len(trajectory_poses)))
        ax.view_init(elev=elev, azim=azim)  # 改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        if title != None:
            ax.set_title(title)
        if fig_name != None:
            plt.savefig(fig_name)
        plt.show()