import os
import shutil
import sys

sys.path.append(os.path.realpath('.'))
sys.path.append(os.path.realpath('../const'))


# 基于机器学习完成的模型对图像进行识别
# 在此之前，要完成模型的训练，模型的训练可以参考官方文档 [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)

# 必要的库的引入
import numpy as np
import os
import tensorflow as tf
import glob
from distutils.version import StrictVersion
from io import StringIO, BytesIO
from PIL import Image
from matplotlib import pyplot as plt

# 相比于官方代码，这里的模块导入代码有所改动
from object_detection.utils import ops as utils_ops

# Tensorflow 版本检测
if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
    raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

# Object detection imports - 对象识别主要模块的导入
# Here are the imports from the object detection module.
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from const.config import PATH_LABEL_MAP,TRAIN_MODEL_PATH

# Model preparation - 预训练模型准备

# Variables - 变量定义
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_FROZEN_GRAPH` to point to a new .pb file.

# # 组件标签
# # List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = PATH_LABEL_MAP

# Loading label map - 加载标签的映射
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)


# Helper code - 帮助函数：将图片数据转换成 numpy 库定义的数组
def load_image_into_numpy_array(image):
    # 这里是针对 PNG 图片进行处理
    if image.format == "PNG":
        image = image.convert('RGB')
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# Detection - 检测部分的代码

# 图片设置暂时注释，目前并未生成图片
# Size, in inches, of the output images - 设置输出的图片尺寸，经过多次调试证明，这个尺寸还比较合适
# IMAGE_SIZE = (48, 32)

# 识别单张图片并返回数据
def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)
            if 'detection_masks' in tensor_dict:
                # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                    detection_masks, detection_boxes, image.shape[1], image.shape[2])
                detection_masks_reframed = tf.cast(
                    tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                    detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict


def toImage(output_dict, image_np, key, path_detect):
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        instance_masks=output_dict.get('detection_masks'),
        use_normalized_coordinates=True,
        line_thickness=8)
    # 画图输出
    plt.imsave(path_detect + '/' + key + '-dec.jpg', image_np)


def toJSON(output_dict, image_np, key, algorithm):
    # 过滤出有效的识别结果，这里设置的分数大于等于 0.1 是经过实验的
    # 这个值需要进一步实验
    cut_off_scores = len(list(filter(lambda x: x >= 0.1, output_dict['detection_scores'])))

    # 每个图片识别结果的数据结构
    item = {
        "pageName": key,
        "boxes": []
    }

    # 矩形面积
    boxes_data = []

    # 遍历有效数据列表
    for index in range(cut_off_scores):
        # 从数据中取值
        score = output_dict['detection_scores'][index]
        class_name = output_dict['detection_classes'][index]
        # Assumption: ymin, xmin, ymax, xmax:
        boxes = output_dict['detection_boxes'][index]
        ymin = float(boxes[0])
        xmin = float(boxes[1])
        ymax = float(boxes[2])
        xmax = float(boxes[3])
        size = (ymax - ymin) * (xmax - xmin)
        if class_name in category_index:
            position = {
                'card': category_index[class_name]['name'],
                'yMin': ymin,
                'xMin': xmin,
                'yMax': ymax,
                'xMax': xmax
            }

            boxes_data.append(dict(position, **{
                'size': size
            }))

            # 拼装数据
            item['boxes'].append(dict(position, **{
                'score': float(score)
            }))

    return item


# 检测图片功能的入口
def detection():
    algorithm = 'faster'
    PATH_TO_FROZEN_GRAPH = TRAIN_MODEL_PATH+'/frozen_inference_graph.pb'

    # 检测输出图片路径
    TRAIN_DETECTIONS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../train_detections/' + algorithm))

    if os.path.exists(TRAIN_DETECTIONS_PATH) is False:
        os.makedirs(TRAIN_DETECTIONS_PATH)
    # 从本地获取一个冻结的 Tensorflow 模型到内存中
    # Load a (frozen) Tensorflow model into memory.

    detection_graph = tf.Graph()
    # pylint: disable=not-context-manager
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    result = []

    imgs_path = glob.glob(os.path.join(TRAIN_DETECTIONS_PATH, '*.jpg'))
    # 对每个文件进行识别
    for path in imgs_path:
        with tf.gfile.GFile(path, 'rb') as fid:
            # 将图片文件转换成 Image 对象
            image = Image.open(BytesIO(fid.read()))
            key = os.path.split(path)[-1].replace('.jpg', '')
            # 获取图片识别数据的部分
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = run_inference_for_single_image(image_np_expanded, detection_graph)

            # 保存图片
            toImage(output_dict, image_np, key, TRAIN_DETECTIONS_PATH)
            # 每个图片识别结果的数据结构
            item = toJSON(output_dict, image_np, key, algorithm)
            print('item', item)
            result.append(item)


if __name__ == '__main__':
    detection()
