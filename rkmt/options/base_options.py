import argparse


class BaseOptions:
    def __init__(self):
        self.initialized = False
        self.parser = None
        self.opt = None

    def initialize(self, parser):
        parser.add_argument('--name', type=str)
        self.initialized = True
        return parser

    def gather_options(self, additional_args=None, estimator_cls=None):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        else:
            parser = self.parser

        # get the basic options
        opt, unknown = parser.parse_known_args(args=additional_args)

        # modify model-related parser options
        if estimator_cls is not None:
            parser = estimator_cls.modify_commandline_options(
                parser, additional_args)

        opt, unknown = parser.parse_known_args(args=additional_args)
        opt = parser.parse_args(args=additional_args)

        self.parser = parser

        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = f'\t[default: {str(default)}]'
            message += f'{str(k):>25}: {str(v):<30}{comment}\n'
        message += '----------------- End -------------------'
        print(message)

    def parse(self, additional_args=None):
        opt = self.gather_options(additional_args=additional_args)
        self.opt = opt
        return self.opt
