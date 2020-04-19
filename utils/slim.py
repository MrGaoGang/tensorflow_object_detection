import os
import sys

# 将 slim 添加到查找路径中
def add_slim_to_path():
    slim_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../slim'))
    sys.path.append(slim_path)