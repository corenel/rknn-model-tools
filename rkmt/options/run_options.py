from rkmt.options.base_options import BaseOptions


class RunOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('-m', '--model_path', type=str)
        parser.add_argument('-i', '--input', type=str)
        parser.add_argument('--results_dir',
                            type=str,
                            default='./results/',
                            help='saves results here.')
        parser.add_argument('-t',
                            '--target',
                            type=str,
                            default='rk1808',
                            choices=['rk1808', 'pc'])
        parser.add_argument('-p', '--print_profiling', action='store_true')
        return parser
