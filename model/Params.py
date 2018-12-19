import json
import os

class Params(object):
    def __init__(self, params_dict:dict=None):
        self.params = dict()
        if params_dict and len(params_dict) > 0:
            self.params = params_dict.copy()

    def save(self, json_file_path:str):
        Params.__to_json_file(json_file_path=json_file_path, parms=self.params)

    @staticmethod
    def create(json_file_path: str):
        if not json_file_path or not os.path.exists(json_file_path):
            raise ValueError("__from_json_file - json_file_path does not exist")

        import json
        with open(json_file_path, 'r') as fp:
            params = json.load(fp=fp)
        return Params(params_dict=params)

    def __iter__(self):
        return iter(self.params)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, key):
        return self.params[key]

    def __setitem__(self, key, value):
        self.params[key] = value

    def __contains__(self, item):
        return item in self.params

    @staticmethod
    def __to_json_file(json_file_path: str, parms: dict):
        if not json_file_path:
            raise ValueError("__to_json_file - json_file_path does not exist")

        import json

        with open(json_file_path, 'w') as f:
            # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
            d = {k: float(v) for k, v in d.items()}
            json.dump(parms, fp=json_file_path, indent=2)
