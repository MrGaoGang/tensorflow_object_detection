import tensorflow as tf
import os
import shutil
import sys

sys.path.append(os.path.abspath("."))
sys.path.append(os.path.abspath("../const"))

from const.config import TRAIN_MODEL_PATH, TRAIN_DATA_PATH, PATH_FASTER_CONFIG

from utils.slim import add_slim_to_path


def get_checkpoint_path():
    CHECKPOINT_PATH = os.path.join(TRAIN_DATA_PATH, 'checkpoint')

    with open(CHECKPOINT_PATH, 'r') as checkpoint_file:
        line = checkpoint_file.readline()
        file_name = line.split('/').pop().replace('"', '').replace('\n', '')
        file_path = os.path.join(TRAIN_DATA_PATH, file_name)
        return file_path


def export(*args):
    '''
    供 Python 调用的方法
    '''
    print('==========')
    from object_detection.export_inference_graph import main

    ALGORITHM_CONFIG_PATH = PATH_FASTER_CONFIG

    if os.path.exists(TRAIN_MODEL_PATH):
        shutil.rmtree(TRAIN_MODEL_PATH)
    os.makedirs(TRAIN_MODEL_PATH)

    argv = [
        '--input_type=image_tensor',
        '--pipeline_config_path=' + ALGORITHM_CONFIG_PATH,
        '--trained_checkpoint_prefix=' + get_checkpoint_path(),
        '--output_directory=' + TRAIN_MODEL_PATH
    ]

    print(argv)

    tf.app.run(
        main=main,
        argv=argv
    )


if __name__ == '__main__':
    add_slim_to_path()
    export()
