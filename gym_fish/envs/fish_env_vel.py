from sys import path
from typing import Any, Dict, Tuple
from .coupled_env import coupled_env
import gym_fish.envs.lib.pyflare as fl
import gym_fish.envs.py_util.np_util as np_util
import numpy as np
import os
import math


def get_full_path(asset_path):
    if asset_path.startswith("/"):
        full_path = asset_path
    else:
        full_path = os.path.join(os.path.dirname(__file__), asset_path)
    if not os.path.exists(full_path):
        raise IOError("File %s does not exist" % full_path)
    return full_path


class FishEnvVel(coupled_env):
    def __init__(self,
                 control_dt=0.2,
                 wv=1.0,
                 wd=0.5,
                 wa=0.5,
                 action_max=5,
                 max_time=10,
                 done_dist=0.1,
                 theta=np.array([-45, 45]),
                 random_vel=np.array([0, 0.3]),
                 data_folder="./data/vis_data/",
                 fluid_json: str = '../assets/fluid/fluid_param_0.5.json',
                 rigid_json: str = '../assets/rigid/rigids_4_30_new.json',
                 gpuId: int = 0,
                 couple_mode: fl.COUPLE_MODE = fl.COUPLE_MODE.TWO_WAY) -> None:
        fluid_json = get_full_path(fluid_json)
        rigid_json = get_full_path(rigid_json)
        if data_folder.startswith("/"):
            data_folder = os.path.abspath(data_folder)
        else:
            data_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), data_folder))
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        self.wv = wv
        self.wd = wd
        self.wa = wa
        self.done_dist = done_dist
        self.random_vel = random_vel
        self.theta = theta / 180.0 * math.pi
        self.control_dt = control_dt
        self.action_max = action_max
        self.max_time = max_time
        self.training = True
        # use parent's init function to init default env data, like action space and observation space, also init dynamics
        super().__init__(fluid_json, rigid_json, gpuId, couple_mode=couple_mode)
        self.simulator.fluid_solver.set_savefolder(data_folder+'/')

    def _step(self, action) -> None:
        t = 0
        while t < self.control_dt:
            self.simulator.iter(action)
            t = t + self.simulator.dt
            if (self._get_done() and not self.training):
                break

    def _get_reward(self, cur_obs, cur_action):
        agent = self.simulator.rigid_solver.get_agent(0)
        vel = agent.linear_vel
        vel_diff_norm = np.exp(- np.linalg.norm(vel - self.goal_vel) / 0.2)
        vel_reward = self.wv * vel_diff_norm

        action_reward = -np.sum(np.abs(cur_action) ** 0.5) * self.wa
        cur_dir = vel / (np.linalg.norm(vel) + 1e-6)

        ori_reward = self.wd * (np.dot(cur_dir, self.goal_dir) + 1) / 2

        total_reward =  action_reward + ori_reward + vel_reward

        info = { "vel_reward": vel_reward, "action_reward": action_reward,
                'ori_reward': ori_reward}
        return min(max(-5, total_reward), 5), info

    def _get_done(self) -> bool:
        done = False
        done = done or self.simulator.time > self.max_time
        done = done or (not np.isfinite(self._get_obs()).all())
        return done

    def _get_obs(self) -> np.array:
        agent = self.simulator.rigid_solver.get_agent(0)
        obs = np.concatenate(
            (
                np.array([self.angle_to_target/math.pi]),
                self.dvel_local,
                agent.positions / 0.52,
                agent.velocities / 10,
            ), axis=0)
        return obs

    def _update_state(self):
        agent = self.simulator.rigid_solver.get_agent(0)
        self.body_xyz = agent.com
        vel = agent.linear_vel
        # update local matrix
        x_axis = agent.fwd_axis
        y_axis = agent.up_axis
        z_axis = agent.right_axis
        self.world_to_local = np.linalg.inv(np.array([x_axis, y_axis, z_axis]).transpose())
        self.rpy = np.arccos(np.array([x_axis[0], y_axis[1], z_axis[2]]))

        self.angle_to_target = np.arccos(np.dot(x_axis,self.goal_dir))
        if np.dot(self.goal_dir, agent.right_axis) < 0:
            self.angle_to_target = -self.angle_to_target
        # in local coordinate
        self.dvel_local = np.dot(self.world_to_local,np.transpose(vel-self.goal_vel))

    def set_task(self, theta, vel):
        agent = self.simulator.rigid_solver.get_agent(0)
        self.init_pos = agent.com
        self.goal_dir = np.array([math.cos(theta), 0, math.sin(theta)])
        self.goal_pos = self.init_pos + 2 * self.goal_dir
        self.goal_vel = self.goal_dir * vel

    def _reset_task(self):
        theta = self.np_random.uniform(self.theta[0],self.theta[1])
        vel = self.np_random.uniform(self.random_vel[0],self.random_vel[1],size=1)[0]
        self.set_task(theta,vel)

    def reset(self) -> Any:
        self.resetDynamics()
        self._reset_task()
        self._update_state()
        self.trajectory_points = []
        return self._get_obs()

        # def plot3d(self, title=None, fig_name=None, elev=45, azim=45):
        #     path_points = np.array([
        #         self.init_pos * (1.0 - t) + self.goal_pos * t for t in np.arange(0.0, 1.0, 1.0 / 100)
        #
        #     ])
        #     trajectory_points = self.trajectory_points
        #     ax = plt.figure().add_subplot(111, projection='3d')
        #     X = path_points[:, 0]
        #     Y = path_points[:, 1]
        #     Z = path_points[:, 2]
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('z')
        #     ax.set_zlabel('y')
        #     max_range = np.array([X.max() - X.min(), Y.max() - Y.min(), Z.max() - Z.min()]).max() / 2.0
        #     mid_x = (X.max() + X.min()) * 0.5
        #     mid_y = (Y.max() + Y.min()) * 0.5
        #     mid_z = (Z.max() + Z.min()) * 0.5
        #     ax.set_xlim(mid_x - max_range, mid_x + max_range)
        #     ax.set_ylim(mid_z - max_range, mid_z + max_range)
        #     ax.set_zlim(mid_y - max_range, mid_y + max_range)
        #     #         if path_points!=None:
        #     #             ax.scatter3D(xs=[x.data[0] for x in path_points], zs=[x.data[1] for x in path_points], ys=[x.data[2] for x in path_points],c='g')
        #     ax.scatter3D(xs=X, zs=Y, ys=Z, c='g')
        #     if trajectory_points != None:
        #         ax.scatter3D(xs=[x[0] for x in trajectory_points],
        #                      zs=[x[1] for x in trajectory_points],
        #                      ys=[x[2] for x in trajectory_points],
        #                      c=[[0, 0, i / len(trajectory_points)] for i in range(len(trajectory_points))])
        #     ax.view_init(elev=elev, azim=azim)  # 改变绘制图像的视角,即相机的位置,azim沿着z轴旋转，elev沿着y轴
        #     ax.set_xlabel('x')
        #     ax.set_ylabel('z')
        #     ax.set_zlabel('y')
        #     if title != None:
        #         ax.set_title(title)
        #     if fig_name != None:
        #         plt.savefig(fig_name)
        #     plt.show()





