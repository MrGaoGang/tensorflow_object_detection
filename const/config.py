import os

# 算法配置地址
PATH_FASTER_CONFIG = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../config/faster_rcnn_inception_v2_pets.config'))
# 训练集训练后模型数据
TRAIN_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_data/faster'))
# 模型训练
TRAIN_MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_model/faster'))
# 测试集的地址
DETECTION_IMG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../test/faster'))

# label map的地址
PATH_LABEL_MAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config/labelmap.pbtxt'))
# 验证集的地址
VALID_DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../valid_assets/faster'))
