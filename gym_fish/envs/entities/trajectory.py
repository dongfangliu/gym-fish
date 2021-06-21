

from gym_fish.envs.py_util import flare_util
import numpy as np
class pose:
    def __init__(self,position:np.array = np.array([0,0,0]),orientation:np.array = np.array([0,0,0]) ) -> None:
        self.position = position
        # specific representation to be determined
        self.orientation = orientation

class trajectory:
    def __init__(self,path_data:flare_util.path_data) -> None:
        self._traj = path_data.trajectory
        self._points = [p.data for p in path_data.trajectory.points]
    @property
    def points(self):
        return self._points
    def get_ref_pose(self,x:float,y:float,z:float) -> pose:
        t  = self.parameterize(x,y,z)
        p = self._traj.getPose(t)
        return pose(p.getPosition(),p.getOrientation())
    def get_curvature(self,x:float,y:float,z:float)->np.array:
        return self._traj.getCurvature(np.array([x,y,z]))

    def parameterize(self,x:float,y:float,z:float) -> float:
        return self._traj.getReferencePose(np.array([x,y,z]))

    