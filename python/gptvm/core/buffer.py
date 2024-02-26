from gptvm.core.device import *

from ctypes import *


class GVBuffer(object):
    """
    class GVBuffer reprensents data buffer on a device.
    """

    def __init__(self, data: bytes | list[int, int], device=GVDevice(0)):
        """GVDevice constructor.

        Args:
            data: bytes or address and size list of data.
            device: Device which the buffer is on.
        """
        if type(data) == bytes:
            self.__address = cast(cast(data, POINTER(c_char)), c_void_p).value
            self.__size = len(data)
            self.__data = data
        else:
            # TODO: Need to support remote CPU and GPU memory
            if True:  # self.device == GVDevice(0):
                buffer = (c_char * data[1]).from_address(data[0])
                self.__data = bytes(buffer)
                self.__address = cast(
                    cast(self.__data, POINTER(c_char)), c_void_p
                ).value
                self.__size = len(self.__data)
            else:
                self.__address = data[0]
                self.__size = data[1]
                self.__data = b""

        self.__device = device

    @property
    def device(self):
        return self.__device

    @property
    def address(self):
        return self.__address

    @property
    def size(self):
        return self.__size

    @property
    def data(self):
        return self.__data
