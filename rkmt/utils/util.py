import os


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def check_success(ret, msg):
    if ret != 0:
        print(msg)
        exit(ret)
