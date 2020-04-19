import tensorflow as tf
import os
import shutil
import sys

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.realpath('../const'))

from const.config import VALID_DATA_PATH,TRAIN_MODEL_PATH,PATH_FASTER_CONFIG
from utils.slim import add_slim_to_path

def eval():
    '''
    供 Python 调用的方法
    '''
    import object_detection.legacy.eval
    from object_detection.legacy.eval import main

    ALGORITHM_CONFIG = PATH_FASTER_CONFIG

    if os.path.exists(VALID_DATA_PATH):
        shutil.rmtree(VALID_DATA_PATH)

    os.makedirs(VALID_DATA_PATH)

    argv = [
        '--logtostderr',
        '--checkpoint_dir=' + TRAIN_MODEL_PATH,
        '--pipeline_config_path=' + ALGORITHM_CONFIG,
        '--eval_dir=' + VALID_DATA_PATH
    ]

    print(argv)

    tf.app.run(
        main=main,
        argv=argv
    )

if __name__ == '__main__':
    add_slim_to_path()
    eval()
