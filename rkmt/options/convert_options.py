from rkmt.options.base_options import BaseOptions


class ConvertOptions(BaseOptions):
    """Arguments parser for model conversion."""
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--platform',
                            type=str,
                            help='deep learning framework')
        # model config
        parser.add_argument('--channel_mean_value',
                            type=str,
                            help='mean and scale parameters for pre-process')
        parser.add_argument(
            '--reorder_channel',
            type=str,
            help='the permutation order of the dimensions of input image')

        # model loading
        parser.add_argument('--model_file_path',
                            type=str,
                            help='the path of model file')
        parser.add_argument('--graph_file_path',
                            type=str,
                            help='the path of model graph definition file')
        parser.add_argument('--inputs',
                            nargs='+',
                            type=str,
                            help='the input nodes of model')
        parser.add_argument('--outputs',
                            nargs='+',
                            type=str,
                            help=' the output nodes of model')
        parser.add_argument(
            '--input_size_list',
            nargs='+',
            type=str,
            help=
            'the size and number of channels of the input tensors corresponding to the input nodes'
        )

        # model building
        parser.add_argument(
            '--dataset_file_path',
            type=str,
            help='a input data set for rectifying quantization parameters')
        parser.add_argument(
            '--dataset_for_analysis_file_path',
            type=str,
            help=
            'a input data set for analysing quantization accuracy (need to contain one line)'
        )
        parser.add_argument(
            '--no_pre_compile',
            action='store_true',
            help='whether or not to pre-compile model for specific hardware')
        parser.add_argument('--no_quantization',
                            action='store_true',
                            help='whether or not to quantize the model')
        parser.add_argument('--output_path',
                            type=str,
                            help='path to converted model')

        # additional flags
        parser.add_argument('-v',
                            '--verbose',
                            action='store_true',
                            help='print log form RKNN')
        parser.add_argument(
            '-a',
            '--analyse_accuracy',
            action='store_true',
            help='whether or not to analysis quantization accuracy')

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
