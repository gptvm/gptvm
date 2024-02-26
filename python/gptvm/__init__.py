import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "lib"))

from libgptvm import *

from gptvm.core import *
GV_CPU=GVDeviceType.GV_CPU
GV_NV_GPU=GVDeviceType.GV_NV_GPU
GV_HW_ASCEND=GVDeviceType.GV_HW_ASCEND
