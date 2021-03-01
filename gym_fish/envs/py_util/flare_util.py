from ..lib import pyflare as fl
from . import json_util
from . import trajectory_util
from pathlib import Path


#############################################################################################
################################ START  WRITE JSON WRAPPER #################################
#############################################################################################
class fluid_param(json_util.json_support):
    normal_attrs = ['x0', 'y0', 'z0', 'width', 'height', 'depth', 'N', 'l0p', 'u0k', 'u0p', 'rou0p', 'visp','pml_width','slip_ratio']
    enum_attrs = ['setup_mode']

    def __init__(self):
        super().__init__()
        self.data = fl.make_simParams()

    def from_dict(self, d: dict):
        for attr in self.normal_attrs :
            if hasattr(self.data, attr) and (attr in d.keys()):
                setattr(self.data, attr, d[attr])
        if hasattr(self.data,'setup_mode')and ('setup_mode' in d.keys()):
            self.data.setup_mode =fl.SETUP_MODE(d['setup_mode'])

    def to_dict(self):
        d = {}
        for attr in self.normal_attrs:
            if hasattr(self.data, attr):
                d[attr] = getattr(self.data, attr)
        for attr in self.enum_attrs:
            if hasattr(self.data, attr):
                d[attr] = int(getattr(self.data, attr))
        return d


class path_param(json_util.json_support):
    def __init__(self, source_file: str=None):
        super().__init__()
        self.source_file = source_file
        if source_file!=None:
            self.points = trajectory_util.trajectoryPoints_file(self.source_file)
            self.path_sample_num=100
        else:
            self.points=[]
            self.path_sample_num=0


    def setPoints(self, points: [fl.trajectoryPoint3d]):
        self.source_file = None
        self.points = points

    def to_dict(self) -> dict:
        d = {'path_sample_num': self.path_sample_num}
        if self.source_file != None:
            d['source_file'] = self.source_file
        else:
            d['points'] = [[x.data[0], x.data[1], x.data[2]] for x in self.points]
        return d

    def from_dict(self, d: dict):
        if 'path_sample_num' in d.keys():
            self.path_sample_num = d['path_sample_num']
        if 'source_file' in d.keys():
            path_skeletonFile = Path(d['source_file']).resolve()
            self.source_file = str(path_skeletonFile)
            self.points = trajectory_util.trajectoryPoints_file(self.source_file)
        else:
            for p in d['points']:
                point = fl.make_tpPoint()
                point.data = p
                self.points.append(point)


class path_data(json_util.json_support):

    def __init__(self, path_setting: path_param=None):
        super().__init__()
        self.trajectory = fl.make_trajectory()
        self.path_setting = path_setting
        if self.path_setting!=None:
            self.setPoints(self.path_setting.points, self.path_setting.path_sample_num)

    def setPoints(self, points, sample_num):
        self.points = points
        self.trajectory.setPoints(self.points)
        self.trajectory.fit()
        self.trajectory.sample(sample_num)

    def to_dict(self) -> dict:
        return self.path_setting.to_dict()

    def from_dict(self, d: dict):
        self.path_setting = path_param()
        self.path_setting.from_dict(d)
        self.setPoints(self.path_setting.points, self.path_setting.path_sample_num)


class skeleton_param(json_util.json_support):
    def __init__(self, skeleton_file: str="", sample_num: int=5000, offset_pos: [float, float, float] = [0, 0, 0],
                 offset_rotation: [float, float, float] = [0, 0, 0]):
        super().__init__()
        self.skeleton_file = str(Path(skeleton_file).resolve())
        self.sample_num = sample_num
        self.offset_pos = offset_pos
        self.offset_rotation = offset_rotation
        self.offset_scale = [1, 1, 1]

    def to_dict(self) -> dict:	
        return self.__dict__

    def from_dict(self, d: dict):
        self.__dict__ = d
        self.skeleton_file = str(Path(self.skeleton_file ).resolve())
        


class skeleton_data(json_util.json_support):

    def __init__(self, skeleton_setting: skeleton_param=None,gpuId:int = 0):
        super().__init__()
        self.skeleton_setting = skeleton_setting
        self.skeleton=None
        self.dynamics =None
        self.gpuId = gpuId
        if skeleton_setting!=None:
            self.init_from_setting()
    def init_from_setting(self):
        self.skeleton = fl.skeletonFromJson(self.skeleton_setting.skeleton_file,self.gpuId)
        self.dynamics = fl.make_skDynamics(self.skeleton,
                                            self.skeleton_setting.sample_num,
                                            self.gpuId,
                                           self.skeleton_setting.offset_pos,
                                            self.skeleton_setting.offset_rotation,
                                            self.skeleton_setting.offset_scale
                                            )
    def to_dict(self) ->dict:
        if self.skeleton_setting!=None:
            return self.skeleton_setting.to_dict()
        else:
            return {}
    def from_dict(self,d:dict):
        self.skeleton_setting = skeleton_param()
        self.skeleton_setting.from_dict(d)
        self.init_from_setting()



class rigid_data(json_util.json_support):
    def __init__(self, gravity=None, skeletons=None,gpuId:int=0):
        super().__init__()
        self.gpuId = gpuId
        if skeletons is None:
            skeletons = []
        self.skeletons = skeletons
        if gravity is None:
            gravity = [0, 0, 0]
        self.gravity = gravity
        self.rigidWorld = fl.make_skWorld(self.gravity)
        for skeleton in self.skeletons:
            self.rigidWorld.addSkeleton(skeleton.dynamics)
    def to_dict(self) ->dict:
        d = {}
        d['skeletons'] = [self.skeletons[i].to_dict() for i in range(len(self.skeletons))]
        d['gravity'] = self.gravity
        return d
    def from_dict(self,d:dict):
        if  'gravity' in d.keys():
            self.gravity = d['gravity']
        else:
            self.gravity=[0,0,0]
        self.rigidWorld.reset()
        self.rigidWorld.setGravity(self.gravity)
        if 'skeletons' not in d.keys():
            return
        self.skeletons.clear()
        for skeleton_dict in d['skeletons']:
            sk  = skeleton_data(None,self.gpuId)
            sk.from_dict(skeleton_dict)
            self.rigidWorld.addSkeleton(sk.dynamics)
            self.skeletons.append(sk)



