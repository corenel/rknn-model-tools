import argparse


class BaseEngine(object):
    def __init__(self, opt) -> None:
        self.opt = opt

    @staticmethod
    def modify_commandline_options(parser: argparse.ArgumentParser,
                                   additional_args=None):
        return parser
