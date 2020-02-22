from rkmt.engines import Estimator


class DummyEstimator(Estimator):
    def pre_process(self, inputs):
        pass

    def post_process(self, outputs):
        pass

    def display_results(self, inputs, results):
        pass
