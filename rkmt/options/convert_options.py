from rkmt.options.base_options import BaseOptions


class ConvertOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--platform',
                            type=str,
                            help='deep learning framework')
        # model config
        parser.add_argument('--channel_mean_value', type=str)
        parser.add_argument('--reorder_channel', type=str)

        # model loading
        parser.add_argument('--model_file_path', type=str)
        parser.add_argument('--graph_file_path', type=str)
        parser.add_argument('--inputs', nargs='+', type=str)
        parser.add_argument('--outputs', nargs='+', type=str)
        parser.add_argument('--input_size_list', nargs='+', type=str)

        # model building
        parser.add_argument('--dataset_file_path', type=str)
        parser.add_argument('--no_pre_compile', action='store_true')
        parser.add_argument('--no_quantization', action='store_true')
        parser.add_argument('--output_path', type=str)

        # additional flags
        parser.add_argument('-v', '--verbose', action='store_true')
        parser.add_argument('-a', '--analyse_accuracy', action='store_true')

        return parser

    def parse(self, additional_args=None, estimator_cls=None):
        opt = super().parse(additional_args, estimator_cls)

        assert len(opt.channel_mean_value.split(',')) in (4, 5)
        assert len(opt.reorder_channel.split(',')) == 3
        opt.channel_mean_value = opt.channel_mean_value.replace(',', ' ')
        opt.reorder_channel = opt.reorder_channel.replace(',', ' ')

        if opt.platform == 'tensorflow':
            assert len(opt.inputs) == len(opt.input_size_list)
        if opt.input_size_list is not None and len(opt.input_size_list) > 0:
            opt.input_size_list = [
                list(map(int, input_size.split('x')))
                for input_size in opt.input_size_list
            ]

        self.opt = opt
        return self.opt
