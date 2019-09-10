import json
from os.path import join as pjoin


class Parameters:
    """Class that loads hyperparameters from a json file.

    From :
    - https://github.com/cs230-stanford/cs230-code-examples/blob/master/pytorch/vision/utils.py
    
    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, params=None, path=None):
        if params is not None:
            self.__dict__.update(params)
        elif path is not None:
            self.update(path)
        else:
            raise Exception("params and path at None ! One of them must be not None.")

    def save(self, path):
        """Saves parameters to a json file"""
        with open(pjoin(path, "params.json"), "w") as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, path):
        """Loads parameters from json file"""
        with open(pjoin(path, "params.json")) as f:
            params = json.load(f)
            params[
                list(self.__dict__.keys())[list(self.__dict__.values()).index(path)]
            ] = path
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__
