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
        ret = self.rknn.init_runtime(target=self.get_target())
        check_success(ret, 'Init runtime environment failed')
        self.timer.log_and_restart('init runtime')

    @abstractmethod
    def pre_process(self, inputs):
        """
        Pre-process input numpy image to be fed into model

        :param inputs: numpy image or other inputs
        :return: processed inputs
        """
        return inputs

    @abstractmethod
    def post_process(self, outputs):
        """
        Post-process the output of model

        :param outputs: output from model inference
        :return: meaningful results
        """
        return outputs

    def inference(self, inputs):
        """
        Do model inference and get outputs

        :param inputs: inputs
        :return: results form model
        """
        self.timer.restart()
        inputs = self.pre_process(inputs)
        self.timer.log_and_restart('pre-process')
        outputs = self.rknn.inference(inputs)
        self.timer.log_and_restart('inference')
        results = self.post_process(outputs)
        self.timer.log_and_restart('post-process')
        return results

    @abstractmethod
    def display_results(self, inputs, results):
        """
        Display results on input numpy image

        :param inputs: input numpy image
        :param results: results from model
        :return: processed numpy image
        """
        return inputs

    def print_profiling(self) -> None:
        """
        Print profiling information
        """
        self.timer.print_log()

    def get_model_path(self):
        """
        Get file path of RKNN model

        :return: file path of RKNN model
        """
        return self.opt.model_file_path

    def get_target(self):
        """
        Get target hardware of RKNN model

        :return: target hardware of RKNN model
        """
        return self.opt.target
