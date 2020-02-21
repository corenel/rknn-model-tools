#!/usr/bin/env python3

import os
import sys
import shutil

from rknn.api import RKNN

from rkmt.engines.base import BaseEngine
from rkmt.utils.util import check_success


def convert_model(model_path, out_path, pre_compile):
    exported_rknn_model_path_list = []

    for model_name in model_configs['models']:
        model = model_configs['models'][model_name]

        rknn = RKNN()

        rknn.config(**model['configs'])

        print('--> Loading model...')
        if model['platform'] == 'tensorflow':
            model_file_path = os.path.join(model_path,
                                           model['model_file_path'])
            input_size_list = []
            for input_size_str in model['subgraphs']['input-size-list']:
                input_size = list(map(int, input_size_str.split(',')))
                input_size_list.append(input_size)
            rknn.load_tensorflow(tf_pb=model_file_path,
                                 inputs=model['subgraphs']['inputs'],
                                 outputs=model['subgraphs']['outputs'],
                                 input_size_list=input_size_list)
        elif model['platform'] == 'tflite':
            model_file_path = os.path.join(model_path,
                                           model['model_file_path'])
            rknn.load_tflite(model=model_file_path)
        elif model['platform'] == 'caffe':
            prototxt_file_path = os.path.join(model_path,
                                              model['prototxt_file_path'])
            caffemodel_file_path = os.path.join(model_path,
                                                model['caffemodel_file_path'])
            rknn.load_caffe(model=prototxt_file_path,
                            proto='caffe',
                            blobs=caffemodel_file_path)
        elif model['platform'] == 'onnx':
            model_file_path = os.path.join(model_path,
                                           model['model_file_path'])
            rknn.load_onnx(model=model_file_path)
        else:
            print("platform {} not support!".format(model['platform']))
        print('done')

        if model['quantize']:
            dataset_path = os.path.join(model_path, model['dataset'])
        else:
            dataset_path = './dataset'

        print('--> Build RKNN model...')
        rknn.build(do_quantization=model['quantize'],
                   dataset=dataset_path,
                   pre_compile=pre_compile)
        print('done')

        export_rknn_model_path = "{}.rknn".format(
            os.path.join(out_path, model_name))
        print('--> Export RKNN model to: {}'.format(export_rknn_model_path))
        rknn.export_rknn(export_path=export_rknn_model_path)
        exported_rknn_model_path_list.append(export_rknn_model_path)
        print('done')

    return exported_rknn_model_path_list


class Converter(BaseEngine):
    def __init__(self, opt) -> None:
        super().__init__(opt)
        # Create RKNN object
        self.rknn = RKNN(opt.verbose)

    def convert(self):
        opt = self.opt
        # Config model
        print('--> Configuring model')
        self.rknn.config(channel_mean_value=opt.channel_mean_value,
                         reorder_channel=opt.reorder_channel)
        print('done')

        # Load model
        print('--> Loading model...')
        if opt.platform == 'tensorflow':
            ret = self.rknn.load_tensorflow(
                tf_pb=opt.model_file_path,
                inputs=opt.inputs,
                outputs=opt.outputs,
                input_size_list=opt.input_size_list)
        elif opt.platform == 'tflite':
            ret = self.rknn.load_tflite(model=opt.model_file_path)
        elif opt.platform == 'caffe':
            ret = self.rknn.load_caffe(model=opt.graph_file_path,
                                       proto='caffe',
                                       blobs=opt.model_file_path)
        elif opt.platform == 'onnx':
            ret = self.rknn.load_onnx(model=opt.model_file_path)
        elif opt.platform == 'darknet':
            ret = self.rknn.load_darknet(model=opt.graph_file_path,
                                         weight=opt.model_file_path)
        elif opt.platform == 'pytorch':
            ret = self.rknn.load_pytorch(model=opt.model_file_path,
                                         input_size_list=opt.input_size_list)
        elif opt.platform == 'mxnet':
            ret = self.rknn.load_mxnet(symbol=opt.graph_file_path,
                                       params=opt.model_file_path,
                                       input_size_list=opt.input_size_list)
        else:
            raise RuntimeError('Unsupported platform: {} !'.format(
                opt.platform))
        check_success(ret, 'load model failed.')
        print('done')

        # Build model
        print('--> Building model')
        ret = self.rknn.build(do_quantization=not opt.no_quantization,
                              pre_compile=not opt.no_pre_compile,
                              dataset=opt.dataset_file_path)
        check_success(ret, 'build model failed.')
        print('done')

        # Analyse model
        if not opt.no_quantization and opt.analyse_accuracy:
            print('--> Analyse model')
            analysis_results_dir = '/tmp/accuracy_analysis/{}'.format(opt.name)
            if os.path.exists(analysis_results_dir):
                shutil.rmtree(analysis_results_dir)
            os.makedirs(analysis_results_dir, exist_ok=True)
            ret = self.rknn.accuracy_analysis(inputs=opt.dataset_file_path,
                                              output_dir=analysis_results_dir,
                                              calc_qnt_error=True)
            check_success(ret, 'analyse model failed.')
            print('done')

        # Export RKNN model
        print('--> Export RKNN model')
        ret = self.rknn.export_rknn(opt.output_path)
        check_success(ret, 'export model failed.')
        print('done')


if __name__ == '__main__':
    model_path = sys.argv[1]
    out_path = sys.argv[2]
    pre_compile = sys.argv[3] in ['true', '1', 'True']

    convert_model(model_path, out_path, pre_compile)
