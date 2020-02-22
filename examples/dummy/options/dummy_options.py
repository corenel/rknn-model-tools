from rkmt.options import ConvertOptions, RunOptions


class DummyConvertOptions(ConvertOptions):
    def initialize(self, parser):
        super().initialize(parser)

        # use set_defaults() ot override argument default value
        parser.set_defaults(platform='tensorflow')

        # use add_argument() to add new arguments
        parser.add_argument('--foo', type=str, default='bar')

        return parser


class DummyRunOptions(RunOptions):
    def initialize(self, parser):
        return super().initialize(parser)
