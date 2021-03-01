import json
from pathlib import *
class json_support:
    def __init__(self):
        pass
    def to_dict(self)->dict:
        pass

    def from_dict(self,d:dict):
        pass
    def get_json(self):
        return json.dumps(self.to_dict(), indent=4)
    def to_json(self, filename):
        with open(filename,'w+') as f:
            f.write(self.get_json())

    def from_json(self, filename):
        p  =  Path(filename).resolve()
#         print(str(p))
        with p.open() as f:
            self.from_dict(json.load(f))

