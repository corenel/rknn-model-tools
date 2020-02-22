from rkmt.options.base_options import BaseOptions


class RunOptions(BaseOptions):
    """Arguments parser for model inference."""
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('-m',
                            '--model_file_path',
                            type=str,
                            help='path to RKNN model file')
        parser.add_argument('-i',
                            '--input_file_path',
                            type=str,
                            help='path to input image file')
        parser.add_argument('--results_dir',
                            type=str,
                            default='./results/',
                            help='path to save results')
        parser.add_argument('-t',
                            '--target',
                            type=str,
                            default='rk1808',
                            choices=['rk1808', 'pc'],
                            help='target hardware platform')
        parser.add_argument(
            '-p',
            '--print_profiling',
            action='store_true',
            help='whether or not to print profiling information')
        return parser
