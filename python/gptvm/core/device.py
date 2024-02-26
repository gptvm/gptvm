from functools import singledispatchmethod
from enum import Enum


class GVDeviceType(Enum):
    """
    class GVDeviceType represents type of device.
    """

    GV_CPU = 0
    GV_NV_GPU = 1
    GV_HW_ASCEND = 2


class GVDevice(object):
    """
    class GVDevice reprensents a single device to execute tasks.

    GVDevice uses a unique id to identify a device in the distributed
    heterogenous compute system. GVDevice(0) is reserved as the driver
    device, i.e. the CPU on the head node which starts the application.
    """

    def __init__(self, ids: list | int):
        """GVDevice constructor.

        Args:
            The list includes device id, device type and node id.
        """
        if type(ids) == int:
            self.__global_id = ids
            self.__device_id = ids >> 16 & 0xFFFF
            self.__device_type = GVDeviceType(ids >> 8 & 0xFF)
            self.__node_id = ids & 0xFF
            return

        assert len(ids) == 3, "Size of ids is not equal to 3."
        assert type(ids[0]) == int, "Type error."
        assert type(ids[1]) == GVDeviceType, "Type error."
        assert type(ids[2]) == int, "Type error."

        self.__device_id, self.__device_type, self.__node_id = ids[:3]
        self.__global_id = (
            self.__device_id << 16 | self.__device_type.value << 8 | self.__node_id
        )

    def __str__(self):
        return f"GVDevice(global_id={self.__global_id:08x})"

    def __repr__(self):
        return self.__str__()

    @property
    def device_type(self):
        return self.__device_type

    @property
    def global_id(self):
        return self.__global_id

    @property
    def node_id(self):
        return self.__node_id

    def clear_node(self):
        self.__node_id = 0
        self.__global_id = (
            self.__device_id << 16 | self.__device_type.value << 8 | self.__node_id
        )
