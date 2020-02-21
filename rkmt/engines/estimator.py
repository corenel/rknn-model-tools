from rknn.api import RKNN

from rkmt.engines.base import BaseEngine
from rkmt.utils.timer import Timer
from rkmt.utils.util import check_success
from abc import abstractmethod


class Estimator(BaseEngine):
    def __init__(self, opt) -> None:
        super().__init__(opt)
        self.rknn = RKNN()
        self.timer = Timer()

        # load model
        ret = self.rknn.load_rknn(self.get_model_path())
        check_success(ret, 'Load model failed')
        self.timer.log_and_restart('load model')

        # init runtime
        ret = self.rknn.init_runtime(self.get_target())
        check_success(ret, 'Init runtime environment failed')
        self.timer.log_and_restart('init runtime')

    @abstractmethod
    def pre_process(self, inputs):
        return inputs

    @abstractmethod
    def post_process(self, outputs):
        return outputs

    def inference(self, inputs):
        self.timer.restart()
        inputs = self.pre_process(inputs)
        self.timer.log_and_restart('pre-process')
        outputs = self.rknn.inference(inputs)
        self.timer.log_and_restart('inference')
        results = self.post_process(outputs)
        self.timer.log_and_restart('post-process')
        return results

    @abstractmethod
    def draw_results(self, inputs, results):
        return inputs

    def print_profiling(self):
        self.timer.print_log()

    def get_model_path(self):
        return self.opt.model_file_path

    def get_target(self):
        return self.opt.target
