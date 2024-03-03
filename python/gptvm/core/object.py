from gptvm.core.device import *
from gptvm.core.buffer import *

import torch, ray


class GVObject(object):
    """
    class GVObject represents an abstract data object passed through tasks.

    An GVObject can either be created from raw data buffer, or from an GVTask
    output.
    An task-produced GVObject may or may not contain the actual data content,
    depending on the task execution status.
    """

    def __init__(
        self,
        shape: list[int] = [],
        device: GVDevice = GVDevice([0, GVDeviceType.GV_CPU, 0]),
    ):
        self.__shape = shape
        self.__device = device
        pass

    def setName(self, name):
        self.__name = name

    def getName(self):
        return self.__name

    def getShape(self):
        if len(self.__shape):
            return self.__shape
        return []

    def setDataRef(self, ref):
        self.__data_ref = ref

    def getDataRef(self):
        return self.__data_ref

    def get(self):
        datas, shapes, device = ray.get(self.__data_ref)
        outputs = {}
        for name in datas:
            if type(datas[name]) == list:
                assert datas[name][0]
                assert datas[name][1]
                outputs[name] = GVBuffer([datas[name][0], datas[name][1]], device)
            elif type(datas[name]) == bytes:
                assert len(datas[name])
                outputs[name] = GVBuffer(datas[name], device)
            else:
                assert False
        return outputs, shapes

    @staticmethod
    def create(
        data: bytes | torch.Tensor,
        shape: list[int] = [],
        device: GVDevice = GVDevice(0),
    ):
        """Copy data to local buffer and create an GVObject from local buffer.
        The local buffer will be copied to device when needed.

        Args:
            data: Bytes of data.
            shape: Shape of data in bytes, default is empty list.
            device: Device to locate data buffer.

        Returns:
             Created GVObject.
        """
        if isinstance(data, torch.Tensor):
            object = GVObject(list(data.size()), device)
            object.setDataRef(ray.put(data.numpy().tobytes()))
            return object

        assert type(data) == bytes, "Type error."
        assert type(shape) == list, "Type error."
        assert type(device) == GVDevice, "Type error."
        assert len(data), "Empty data."

        object = GVObject(shape, device)
        object.setDataRef(ray.put(data))
        return object
