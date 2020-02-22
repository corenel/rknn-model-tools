from rkmt.engines import Converter
from options import DummyConvertOptions


def convert():
    # parse model-specific arguments
    opt = DummyConvertOptions().parse()

    # initialize and run conversion
    converter = Converter(opt)
    converter.convert()


if __name__ == '__main__':
    convert()
