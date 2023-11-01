"""!
@file array.py
@brief Contains the dispatching logic for CuPy & NumPy arrays based on the active Task Environment.

This file contains the dispatching logic for CuPy & NumPy arrays based on the active Task Environment. 
This is used to determine which array type to use for a given task. The manual data movement functions `copy` and `clone_here` are defined here.
"""

# This implementation is adapted from the original Parla <0.2 implementation.
# Original author: @arthurp
# In this refactor we have removed MemoryKind types (Fast/Slow) and added more consistent control over the active stream.

from abc import ABCMeta, abstractmethod
from typing import Dict

# FIXME: This load of numpy causes problems if numpy is multiloaded. So this breaks using VECs with parla tasks.
#  Loading numpy locally works for some things, but not for the array._register_array_type call.
import numpy

from parla.common.globals import (
    get_current_context,
    cupy,
    DeviceType,
    CUPY_ENABLED,
    CROSSPY_ENABLED,
    crosspy,
)


class ArrayType(metaclass=ABCMeta):
    @abstractmethod
    def can_assign_from(self, src, dst):
        """
        :param a: An array of self's type.
        :param b: An array of any type.
        :return: True iff `a` supports assignments from `b`.
        """
        raise NotImplementedError(
            "can_assign_from not implemented for type {}".format(type(self).__name__)
        )

    @abstractmethod
    def get_array_module(self, src):
        """
        :param a: An array of self's type.
        :return: The `numpy` compatible module for the array `a`.
        """
        raise NotImplementedError(
            "get_array_module not implemented for type {}".format(type(self).__name__)
        )

    @abstractmethod
    def copy_from(self, src, target_device_id: int):
        """
        :param a: An array of some type.
        :return: A copy of `a` of this type
        """
        raise NotImplementedError(
            "copy_from not implemented for type {}".format(type(self).__name__)
        )

    @abstractmethod
    def copy_into(self, src, dst):
        """
        :param src: An array of some type.
        :param dst: An array of this type.
        :return: A copy of `src` into `dst`.
        """
        raise NotImplementedError(
            "copy_into not implemented for type {}".format(type(self).__name__)
        )


class NumpyArray(ArrayType):
    def can_assign_from(self, src, dst):
        return isinstance(src, numpy.ndarray)

    def get_array_module(self, a):
        return numpy

    def copy_from(self, src, target_device_id: int):
        if isinstance(src, numpy.ndarray):
            return src

        current_context = get_current_context()
        current_device = current_context.devices[0]

        is_gpu = current_device.architecture == DeviceType.CUDA

        if CUPY_ENABLED and isinstance(src, cupy.ndarray):
            if is_gpu and (src.flags["C_CONTIGUOUS"] or src.flags["F_CONTIGUOUS"]):
                dst = cupy.empty_like(src)
                with current_context:
                    target_stream = cupy.cuda.get_current_stream()
                    dst.data.copy_from_async(src.data, src.nbytes, stream=target_stream)
            else:
                dst = cupy.asnumpy(src)

            return dst

        else:
            raise NotImplementedError("Non-ndarray types are not currently supported")

    def copy_into(self, src, dst):
        # TODO: Add contiguous copy into memory buffer directly

        if self.can_assign_from(src, dst):
            dst[:] = src
        elif isinstance(src, cupy.ndarray):
            dst[:] = self.copy_from(src, dst.device.id)
        else:
            raise NotImplementedError("Non-ndarray types are not currently supported")


class CupyArray(ArrayType):
    def can_assign_from(self, src, dst):
        return isinstance(src, cupy.ndarray) and (src.device.id == dst.device.id)

    def get_array_module(self, a):
        return cupy

    def copy_from(self, src, target_device_id: int):
        current_context = get_current_context()
        current_device = current_context.devices[0]

        is_gpu = current_device.architecture == DeviceType.CUDA

        if isinstance(src, cupy.ndarray) or isinstance(src, numpy.ndarray):
            if isinstance(src, cupy.ndarray) and (src.device.id == target_device_id):
                # Do not perform a copy if the array is already on the current device.
                return src

            if src.flags["C_CONTIGUOUS"] or src.flags["F_CONTIGUOUS"]:
                with cupy.cuda.Device(target_device_id) as d:
                    stream = cupy.cuda.Stream(non_blocking=True)
                    event = cupy.cuda.Event()

                    with stream as s:
                        dst = cupy.empty_like(src)

                    if is_gpu:
                        event.record(stream)
                        with current_device:
                            target_stream = cupy.cuda.get_current_stream()
                            target_stream.wait_event(event)

                            memptr = (
                                src.data
                                if isinstance(src, cupy.ndarray)
                                else src.ctypes.data
                            )
                            dst.data.copy_from_async(
                                memptr, src.nbytes, stream=target_stream
                            )
                    else:
                        memptr = (
                            src.data
                            if isinstance(src, cupy.ndarray)
                            else src.ctypes.data
                        )
                        dst.data.copy_from_async(memptr, src.nbytes, stream=stream)
                        stream.synchronize()

                return dst

            if isinstance(src, cupy.ndarray):
                with cupy.cuda.Device(src.device.id):
                    src = cupy.ascontiguousarray(src)

            with cupy.cuda.Device(target_device_id):
                dst = cupy.asarray(src)

            return dst

        else:
            raise NotImplementedError("Non-ndarray types are not currently supported")

    def copy_into(self, src, dst):
        # TODO: Add contiguous copy into memory buffer directly

        if self.can_assign_from(src, dst):
            # FIXME: When is this guaranteed to work for src/dst on different devices
            # Strided access seems to fail (sometimes) on Peer access.
            with cupy.cuda.Device(dst.device.id):
                dst[:] = src
        elif isinstance(src, numpy.ndarray) or isinstance(src, cupy.ndarray):
            temp = self.copy_from(src, dst.device.id)
            with cupy.cuda.Device(dst.device.id):
                dst[:] = temp
                # FIXME: What stream should we use here? How can we synchronize reliably?
        else:
            raise NotImplementedError("Non-ndarray types are not currently supported")


# FIXME: Put this in a better place.
_array_types: Dict[type, ArrayType] = dict()


def _register_array_type(ty, get_memory_impl: ArrayType):
    _array_types[ty] = get_memory_impl


_register_array_type(numpy.ndarray, NumpyArray())

if CUPY_ENABLED:
    _register_array_type(cupy.ndarray, CupyArray())


def can_assign_from(a, b):
    """
    :param a: An array.
    :param b: An array.
    :return: True iff `a` supports assignments from `b`.
    """
    return _array_types[type(a)].can_assign_from(a, b)


def get_array_module(a):
    """
    :param a: A numpy-compatible array.
    :return: The numpy-compatible module associated with the array class (e.g., cupy or numpy).
    """
    return _array_types[type(a)].get_array_module(a)


def is_array(a) -> bool:
    """
    :param a: A value.
    :return: True if `a` is an array of some type known to parla.
    """
    return type(a) in _array_types


def asnumpy(a):
    ar = get_array_module(a)
    if hasattr(ar, "asnumpy"):
        return ar.asnumpy(a)
    else:
        return ar.asarray(a)


def copy(destination, source):
    """
    Copy the contents of `source` into `destination`.
    :param destination: The array to write into.
    :param source: The array to read from or the scalar value to put in destination.
    """
    # FIXME: This doesn't work on PArrays or CrossPyArrays

    try:
        if is_array(source):
            _array_types[type(destination)].copy_into(source, destination)
        else:
            # We assume all non-array types are by-value and hence already exist in the Python interpreter
            # and don't need to be copied.
            destination[:] = source
    except ValueError as e:
        print("Error in copy: ", e)
        raise ValueError(
            "Failed to copy ({} {} to {} {})".format(
                source,
                getattr(source, "shape", None),
                destination,
                getattr(destination, "shape", None),
            )
        ) from e


def clone_here(source, kind=None):
    """
    Create a local copy of `source` stored at the current location.
    :param source: The array to read from.
    """
    if CROSSPY_ENABLED and isinstance(source, crosspy.CrossPyArray):
        # FIXME: This only works on unwrapped CrossPy Arrays
        # FIXME: This doesn't work on noncontigious colorings as it will not copy the index map

        current_context = get_current_context()

        # Our semantics are only defined when the context contains more components than the source array.
        # assert (len(current_context.devices) > len(source.block_view()))

        # We need to copy the array to the current context.
        # From left to right, we copy each "block" of the array to the corresponding device.
        array_list = []
        for i, block in enumerate(source.block_view()):
            with current_context.devices[i]:
                array_list.append(clone_here(block))

        return crosspy.array(array_list, axis=0)

    elif is_array(source):
        # FIXME: This doesn't work on PArrays

        current_content = get_current_context()

        if current_content is None:
            raise ValueError(
                "No device context exists for the current copy. Operation is ambiguous."
            )
        if len(current_content.devices) != 1:
            raise ValueError(
                "Multiple devices exist for the current copy. Operation is ambiguous."
            )

        current_device = current_content.devices[0]

        # FIXME: Make this a property of the device
        if (current_device.architecture == DeviceType.CUDA) and CUPY_ENABLED:
            AType = CupyArray()
        else:
            AType = NumpyArray()

        target_device_id = current_device().device_id
        dst = AType.copy_from(source, target_device_id)

        return dst

    else:
        raise TypeError(
            "ArrayType required, given value of type {}".format(type(source))
        )


def storage_size(*arrays):
    """
    :return: the total size of the arrays passed as arguments.
    """
    return sum(a.size * a.itemsize for a in arrays)
