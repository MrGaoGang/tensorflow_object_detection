import tensorflow as tf
import os
import shutil
import sys

sys.path.append(os.path.abspath('.'))
sys.path.append(os.path.abspath('../utils'))
sys.path.append(os.path.abspath('../const'))

from utils.slim import add_slim_to_path
from const.config import PATH_FASTER_CONFIG, TRAIN_DATA_PATH


def train():
    '''
    供 Python 调用的方法
    '''
    import object_detection.legacy.train
    from object_detection.legacy.train import main

    ALGORITHM_CONFIG_PATH = PATH_FASTER_CONFIG


    if os.path.exists(TRAIN_DATA_PATH):
        shutil.rmtree(TRAIN_DATA_PATH)

    os.mkdir(TRAIN_DATA_PATH)

    argv = [
        '--logtostderr',
        '--pipeline_config_path=' + ALGORITHM_CONFIG_PATH,
        '--train_dir=' + TRAIN_DATA_PATH
    ]

    print(argv)

    tf.app.run(
        main=main,
        argv=argv
    )


if __name__ == '__main__':
    add_slim_to_path()
    train()
