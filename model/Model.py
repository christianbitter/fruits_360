import os
from .Params import Params

class Model(object):
    def __init__(self, params_fp, model, verbose=False):
        self.params_fp = params_fp
        self.verbose = verbose
        self.model_name = model
        self.batch_size = None
        self.no_epochs = None
        self.learning_rate = None
        self.optimizer_name = None

        if self.params_fp and os.path.exists(self.params_fp):
            # now update where possible
            if self.verbose:
                print("_load_params - loading parameters from json")
            self.params = Params.create(json_file_path=self.params_fp)
        else:
            self.params = Params()

    def _load_params(self):
        raise ValueError("_load_params - not implemented")

    def _global_preprocess(self, data, label):
        raise ValueError("_global_preprocess - not implemented")

    def _parse_tfrecord(self, tfrecord_proto):
        raise ValueError("_parse_tfrecord - not implemented")

    def import_data(self):
        raise ValueError("import_data - not implemented")

    def build(self):
        raise ValueError("build - not implemented")

    def prepare(self, model_weights_fp: str=None):
        raise ValueError("prepare - not implemented")

    def summary(self, what: str = 'full'):
        raise ValueError("prepare - summary not implemented")