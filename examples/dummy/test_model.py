import cv2

from rkmt.utils.util import check_success
from engines import DummyEstimator
from options import DummyRunOptions
from tqdm import tqdm


def run():
    # parse model-specific arguments
    opt = DummyRunOptions().parse()
    estimator = DummyEstimator(opt)

    # read frame
    print('--> Reading image')
    image_np = cv2.imread(opt.input_file_path)
    check_success(image_np is None, 'read image failed')
    print('done')

    # inference model
    print('--> Running model')
    results = None
    for _ in tqdm(range(100)):
        results = estimator.inference(image_np)

    # display results
    image_results = estimator.display_results(image_np, results)
    cv2.imwrite('result.png', image_results)

    # print profiling results
    if opt.print_profiling:
        estimator.print_profiling()


if __name__ == '__main__':
    run()
