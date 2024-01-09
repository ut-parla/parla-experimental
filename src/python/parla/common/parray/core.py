from __future__ import annotations
from typing import List, Dict, TYPE_CHECKING, Union, Any

from parla.cython.device import PyCPUDevice
from parla.common.globals import (
    get_current_devices,
    get_scheduler,
    has_environment,
    DeviceType,
    CUPY_ENABLED,
)

from .coherence import MemoryOperation, Coherence, CPU_INDEX
from .memory import MultiDeviceBuffer
from parla.cython.cyparray_state import CyPArrayState
from parla.cython.cyparray import CyPArray

import threading
import numpy

if CUPY_ENABLED:  # if the system has no GPU
    import cupy

    num_gpu = cupy.cuda.runtime.getDeviceCount()
else:
    cupy = numpy
    num_gpu = 0

if TYPE_CHECKING:
    from parla.cython.device import PyDevice

    ndarray = Union[numpy.ndarray, cupy.ndarray]
    SlicesType = Union[slice, int, tuple]


UINT64_LIMIT = 1 << 64 - 1


class PArray:
    """Multi-dimensional array on a CPU or CUDA device.

    This class is a wrapper around :class:`numpy.ndarray` and :class:`cupy.ndarray`,
    It is used to support Parla sheduler optimization and automatic data movement.

    Args:
        array: :class:`cupy.ndarray` or :class:`numpy.array` object

    Note: some methods should be called within the current task context
    """

    _array: MultiDeviceBuffer
    _coherence: Coherence
    _slices: List[SlicesType]
    _coherence_cv: Dict[int, threading.Condition]
    _cy_parray: CyPArray  # the wrapper class of C++ PArray class, which store the size and id
    # the wrapper class of C++ PArrayState class, which store the exist and valid state
    _cyparray_state: CyPArrayState

    def __init__(
        self, array: ndarray, parent: "PArray" = None, slices=None, name: str = "NA"
    ) -> None:
        if parent is not None:  # create a view (a subarray) of a PArray
            # inherit parent's buffer and coherence states
            # so this PArray will becomes a 'view' of its parents
            self._array = parent._array
            self._coherence = parent._coherence
            self._cyparray_state = parent._cyparray_state

            # _slices is a list so subarray of subarray works
            self._slices = parent._slices.copy()  # copy parent's slices list
            # add current slices to the end
            self._slices.append(slices)

            # self._name = parent._name + "::subarray::" + str(slices)
            self._name = parent._name + "::subarray::" + str(self._slices)

            # inherit parent's condition variables
            self._coherence_cv = parent._coherence_cv

            # identify the slices
            self._slices_hash = self._array.get_slices_hash(slices)

            # a unique ID for this subarray
            # which is the combine of parent id and slice hash
            self.parent_ID = parent.ID
            # use a prime number to avoid collision
            # modulo over uint64 to make it compatible with C++ end
            self.ID = (parent.ID * 31 + self._slices_hash) % UINT64_LIMIT

            self.nbytes = parent.nbytes  # the bytes used by the complete array
            self.subarray_nbytes = array.nbytes  # the bytes used by this subarray
            self.parent = parent

            # no need to register again since parent already did that
        else:  # initialize a new PArray
            # init Cython PArrayState
            self._cyparray_state = CyPArrayState()

            # per device buffer of data
            self._array = MultiDeviceBuffer(num_gpu, self._cyparray_state)
            location = self._array.set_complete_array(array)

            # coherence protocol for managing data among multi device
            self._coherence = Coherence(location, num_gpu, self._cyparray_state)

            # no slices since it is a new array rather than a subarray
            self._slices = []

            # a condition variable to acquire when moving data on the device
            self._coherence_cv = {n: threading.Condition() for n in range(num_gpu)}
            self._coherence_cv[CPU_INDEX] = threading.Condition()

            # there is no slices
            self._slices_hash = None

            # use id() of buffer as unique ID
            self.parent_ID = id(self._array)  # no parent
            self.ID = self.parent_ID

            self.nbytes = array.nbytes
            self.subarray_nbytes = self.nbytes  # no subarray

            self.parent = self

            self._name = name

        # record the size in Cython PArray
        scheduler = get_scheduler()
        num_devices = len(scheduler.device_manager.get_all_devices())
        self._cy_parray = CyPArray(
            self,
            self.ID,
            self.parent_ID,
            self.parent,
            self._cyparray_state,
            num_devices,
        )
        self._cy_parray.set_size(self.subarray_nbytes)
        target_dev_id = -1 if isinstance(array, numpy.ndarray) else array.device.id
        if target_dev_id >= 0:
            assert isinstance(array, cupy.ndarray)
        target_global_dev_id = scheduler.device_manager.parrayid_to_globalid(
            target_dev_id
        )
        scheduler.reserve_parray(self._cy_parray, target_global_dev_id)

    # Properties:

    @property
    def name(self):
        if self._name is None:
            return "PArray::" + str(self.ID)
        return self._name

    @property
    def cy_parray(self):
        return self._cy_parray

    @property
    def array(self) -> ndarray:
        """
        The reference to cupy/numpy array on current device.
        Note: should be called within the current task context
        Note: should call A[valid_slices].array instead of A.array,
            when in a tasks which only subarray is auto-moved.
            `valid_slices` is a slices within the auto-moved subarray.
        """
        if self._slices:  # so this is a sub-parray object
            # index into origin array by saved slices
            ret = self._array.get_by_global_slices(
                self._current_device_index, self._slices[0]
            )
            for s in self._slices[1:]:
                ret = ret[s]
            return ret
        else:  # this is a complete copy
            ret = self._array.get(self._current_device_index)

            if isinstance(ret, list):  # get a subarray instead
                raise IndexError(
                    "Current device doesn't have a complete copy of this array"
                )
            return ret

    @property
    def _on_gpu(self) -> bool:
        """
        True if the array is on GPU.
        Note: should be called within the current task context
        """
        return self._current_device_index != CPU_INDEX

    @property
    def _current_device_index(self) -> int:
        """
        -1 if the current device is CPU.
        Otherwise GPU ID.
        If called outside the current task context, return data owner's device_id by default
        """
        device = PArray._get_current_device()
        if device is None:  # not called inside current task
            return self._coherence.owner
        elif isinstance(device, PyCPUDevice):
            return CPU_INDEX
        else:
            # assume GPU here, won't check isinstance(device, PyGPUDevice)
            # to avoid import gpu context, which is slow to setup.
            return device.device.id  # device.device should be a cupy.cuda.Device object

    # Public API:

    def nbytes_at(self, device_id: int):
        """An estimate of bytes used in `device_id` after data is moved there.
        It is neither lower bound nor upper bound.
        """
        if self._slices:
            # there are subarrays no this device
            if isinstance(self._coherence._local_states[device_id], dict):
                # this subarray is already there
                if self._slices_hash in self._coherence._local_states[device_id].keys():
                    return self._array.nbytes_at(device_id)
                else:  # the subarray will be moved to there
                    # add the incoming subarray size
                    return self._array.nbytes_at(device_id) + self.subarray_nbytes
            else:  # there is a complete copy on this device, no need to prepare subarray
                return self.nbytes
        else:
            return self.nbytes

    def exists_on_device(self, device_id: int):
        return self._array._buffer[device_id] is not None

    def update(self, array) -> None:
        """Update the copy on current device.
        Previous copy on other device are lost.
        This will replace the internal buffer and coherence object completly.

        Args:
            array: :class:`cupy.ndarray` or :class:`numpy.array` object

        Note: should be called within the current task context
        Note: data should be put in OUT/INOUT fields of spawn
        Note: should not call this over an sliced array
        """
        this_device = self._current_device_index

        # clean up data
        self._array._buffer = {n: None for n in range(num_gpu)}
        self._array._buffer[CPU_INDEX] = None

        # update it to cpp side
        for n in range(num_gpu):
            self._cyparray_state.set_exist_on_device(n, False)
        self._cyparray_state.set_exist_on_device(CPU_INDEX, False)

        # copy new data to buffer
        if isinstance(array, numpy.ndarray):
            if this_device != CPU_INDEX:  # CPU to GPU
                self._array.set(this_device, cupy.asarray(array))
            else:  # data already in CPU
                self._array.set(this_device, array)
        else:
            if this_device == CPU_INDEX:  # GPU to CPU
                self._array.set(this_device, cupy.asnumpy(array))
            else:  # GPU to GPU
                if int(array.device) == this_device:  # data already in this device
                    self._array.set(this_device, array)
                else:  # GPU to GPU
                    dst_data = cupy.empty_like(array)
                    dst_data.data.copy_from_device_async(array.data, array.nbytes)
                    self._array.set(this_device, dst_data)

        # update size
        self.nbytes = array.nbytes
        self.subarray_nbytes = self.nbytes
        self._cy_parray.set_size(self.nbytes)

        self._slices = []

        # reset coherence
        self._coherence.reset(this_device)

        # update shape
        self._array.shape = array.shape

        if num_gpu > 0:
            cupy.cuda.stream.get_current_stream().synchronize()

    def print_overview(self) -> None:
        """
        Print global overview of current PArray for debugging
        """
        state_str_map = {0: "INVALID", 1: "SHARED", 2: "MODIFIED"}

        print(
            f"---Overview of PArray\n"
            f"ID: {self.ID}, "
            f"Name: {self._name}, "
            f"Parent_ID: {self.parent_ID if self.ID != self.parent_ID else None}, "
            f"Slice: {self._slices[0] if self.ID != self.parent_ID else None}, "
            f"Bytes: {self.subarray_nbytes}, "
            f"Owner: {'GPU ' + str(self._coherence.owner) if self._coherence.owner != CPU_INDEX else 'CPU'}"
        )
        for device_id, state in self._coherence._local_states.items():
            if device_id == CPU_INDEX:
                device_name = "CPU"
            else:
                device_name = f"GPU {device_id}"
            print(f"At {device_name}: ", end="")

            if isinstance(state, dict):
                print(
                    f"state: {[state_str_map[s] for s in list(state.values())]}, including sliced copy:  # states of slices is unordered wrt the below slices"
                )
                for slice, slice_id in zip(
                    self._array._indices_map[device_id],
                    range(len(self._array._indices_map[device_id])),
                ):
                    print(
                        f"\tslice {slice_id} - indices: {slice}, bytes: {self._array._buffer[device_id][slice_id].nbytes}"
                    )
            else:
                print(f"state: {state_str_map[state]}")
        print("---End of Overview")

    # slicing/indexing

    def __getitem__(self, slices: SlicesType) -> PArray | Any:
        if self._slices:  # resolve saved slices first
            ret = self.array[slices]
        else:
            ret = self._array.get_by_global_slices(self._current_device_index, slices)

        # ndarray.__getitem__() may return a ndarray
        if ret is None:
            # when current device has no copy (could happen when a slice annotation is needed at @spawn)
            # get ret from owner, so it could have an estimation of the number of bytes used
            ret = self._array.get_by_global_slices(self._coherence.owner, slices)
            return PArray(ret, parent=self, slices=slices)
        elif isinstance(ret, numpy.ndarray):
            return PArray(ret, parent=self, slices=slices)
        elif isinstance(ret, cupy.ndarray):
            if ret.shape == ():
                return ret.item()
            else:
                return PArray(ret, parent=self, slices=slices)
        else:
            return ret

    def __setitem__(self, slices: SlicesType, value: PArray | ndarray | Any) -> None:
        """
        Acceptable Slices: Slice, Int, tuple of (Slice, Int, List of Int)
        Example:
            A[0]  # int
            A[:]  # slice
            A[0,:,10]  # tuple of int slice int
            A[2:10:2, 0, [1, 3, 5]]  # tuple of slice int list of Int

        Note: `:` equals to slice(None, None, None)
        Note: `None` or tuple of `None` is not acceptable (even if `numpy.ndarray` accept `None`)
        # TODO: support `None` and `ndarray` as slices
        """
        if isinstance(value, PArray):
            value = value.array

        if self._slices:  # resolve saved slices first
            self.array.__setitem__(slices, value)
        else:
            self._array.set_by_global_slices(self._current_device_index, slices, value)

    def evict_all(self) -> None:
        """
        Evict all copies from buffer, and clean all related fields

        Note: this object should not be accessed anymore after called this method
        """
        self._array = None
        self._coherence = None

    def evict(self, device_id: int = None, keep_one_copy: bool = True) -> bool:
        """
        Evict a device's copy and update coherence states.

        Args:
            device_id: if is this not None, data will be moved to this device,
                    else move to current device
            keep_one_copy: if it is True and this is the last copy,
                    write it back to CPU before evict.
        Return:
            true if the copy is evicted successfully
            false if the eviction cannot be performed since it has last copy
                and `keep_one_copy` is false
        """
        if device_id is None:
            device_id = self._current_device_index

        with self._coherence_cv[device_id]:
            operations = self._coherence.evict(device_id, keep_one_copy)
            if operations[0].inst == MemoryOperation.ERROR:
                return False  # cannot perform the eviction
            self._process_operations(operations)

        return True

    # Coherence update operations:

    def _coherence_read(self, device_id: int = None, slices: SlicesType = None) -> None:
        """Tell the coherence protocol a read happened on a device.

        And do data movement based on the operations given by protocol.

        Args:
            device_id: if is this not None, data will be moved to this device,
                    else move to current device
            slices: a slices of the subarray to be manipulated
                    by default equals to None, which means the whole array is manipulated

        Note: should be called within the current task context
        """
        if device_id is None:
            device_id = self._current_device_index

        # update protocol and get operation
        with self._coherence._lock:  # locks involve
            operations = self._coherence.read(device_id, self._slices_hash)
            self._process_operations(operations, slices)

    def _coherence_write(
        self, device_id: int = None, slices: SlicesType = None
    ) -> None:
        """Tell the coherence protocol a write happened on a device.

        And do data movement based on the operations given by protocol.

        Args:
            device_id: if is this not None, data will be moved to this device,
                    else move to current device
            slices: a slices of the subarray to be manipulated
                    by default equals to None, which means the whole array is manipulated

        Note: should be called within the current task context
        """
        if device_id is None:
            device_id = self._current_device_index

        # update protocol and get operation
        with self._coherence._lock:  # locks involve
            operations = self._coherence.write(device_id, self._slices_hash)
            self._process_operations(operations, slices)

    # Device management methods:

    def _process_operations(
        self, operations: List[MemoryOperation], slices: SlicesType = None
    ) -> None:
        """
        Process the given memory operations.
        Data will be moved, and protocol states is kept unchanged.
        """
        for op in operations:
            if op.inst == MemoryOperation.NOOP:
                pass  # do nothing
            elif op.inst == MemoryOperation.LOAD:
                if MemoryOperation.LOAD_SUBARRAY in op.flag:
                    # build slices mapping first
                    self._array.set_slices_mapping(op.dst, slices)

                # check flag to see if dst is current device
                dst_is_current_device = op.flag != MemoryOperation.SWITCH_DEVICE_FLAG

                # copy data
                self._array.copy_data_between_device(
                    op.dst, op.src, dst_is_current_device
                )

                # sync stream before set it as ready, so asyc call is ensured to be done
                if num_gpu > 0:
                    cupy.cuda.stream.get_current_stream().synchronize()
            elif op.inst == MemoryOperation.EVICT:
                scheduler = get_scheduler()
                src_global_dev_id = scheduler.device_manager.parrayid_to_globalid(
                    op.src
                )
                if self._cy_parray.get_num_active_tasks(src_global_dev_id) == 0:
                    # If none of visible tasks will refer this PArray, release
                    # this PArray instance of the source device from the PArray tracker.
                    scheduler.release_parray(self._cy_parray, src_global_dev_id)
                # decrement the reference counter, relying on GC to free the memor
                self._array.clear(op.src)
            elif op.inst == MemoryOperation.ERROR:
                raise RuntimeError("PArray gets an error from coherence protocol")
            else:
                raise RuntimeError(
                    f"PArray gets invalid memory operation from coherence protocol, "
                    f"detail: opcode {op.inst}, dst {op.dst}, src {op.src}"
                )

    @staticmethod
    def _get_current_device() -> PyDevice | None:
        """
        Get current device from task environment.

        Return None if it is not called within the current task context

        Note: should not be called within multi-device task since that is ambigous
        """
        if has_environment():
            return get_current_devices()[0].get_parla_device()
        return None

    def _get_compute_device_for_crosspy(self, index: int = 0):
        """
        Get the active device context for crosspy.
        Returns None if not called from within a task.
        """
        if has_environment():
            current_device = get_current_devices()[index]
            return current_device
        return None

    def _auto_move(self, device_id: int = None, do_write: bool = False) -> None:
        """Automatically move data to current device.

        Multiple copies on different devices will be made based on coherence protocol.

        Args:
            device_id: current device id. CPU use CPU_INDEX as id
            do_write: True if want make the device MO in coherence protocol
                False if this is only for read only in current task

        Note: should be called within the current task context.
        Note: auto-move of subarray's subarray is not supported.
        """
        # this is the view of current array, only data within this range should be moved
        # currently, only use first slices, which means automove of subarray of subarray is not supported
        # TODO: support auto-move subarray of subarray
        slices = None if not self._slices else self._slices[0]

        if do_write:
            self._coherence_write(device_id, slices)
        else:
            self._coherence_read(device_id, slices)

    def _on_same_device(self, other: "PArray") -> bool:
        """
        Return True if the two PArrays are in the same device.
        Note: other has to be a PArray object.
        """
        this_device = self._current_device_index
        return this_device in other._array

    # NumPy/CuPy methods redirection

    def __getattr__(self, item):
        """
        A proxy method that redirect call to methods in :class:`numpy.ndarray` or :class:`cupy.ndarray`
        """
        return getattr(self.array, item)

    # Comparison operators:

    def __lt__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__lt__(y.array)
        else:
            return x.array.__lt__(y)

    def __le__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__le__(y.array)
        else:
            return x.array.__le__(y)

    def __eq__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__eq__(y.array)
        else:
            return x.array.__eq__(y)

    def __ne__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__ne__(y.array)
        else:
            return x.array.__ne__(y)

    def __gt__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__gt__(y.array)
        else:
            return x.array.__gt__(y)

    def __ge__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return x.array.__ge__(y.array)
        else:
            return x.array.__ge__(y)

    # Truth value of an array (bool):

    def __nonzero__(self):
        return PArray(self.array.__nonzero__())

    # Unary operations:

    def __neg__(self):
        return PArray(self.array.__neg__())

    def __pos__(self):
        return PArray(self.array.__pos__())

    def __abs__(self):
        return PArray(self.array.__abs__())

    def __invert__(self):
        return PArray(self.array.__invert__())

    # Arithmetic:

    def __add__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array + y.array)
        else:
            return PArray(x.array + y)

    def __sub__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array - y.array)
        else:
            return PArray(x.array - y)

    def __mul__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array * y.array)
        else:
            return PArray(x.array * y)

    def __matmul__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array @ y.array)
        else:
            return PArray(x.array @ y)

    def __div__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array / y.array)
        else:
            return PArray(x.array / y)

    def __truediv__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array / y.array)
        else:
            return PArray(x.array / y)

    def __floordiv__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__floordiv__(y.array))
        else:
            return PArray(x.array.__floordiv__(y))

    def __mod__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__mod__(y.array))
        else:
            return PArray(x.array.__mod__(y))

    def __divmod__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__divmod__(y.array))
        else:
            return PArray(x.array.__divmod__(y))

    def __pow__(x, y, modulo):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__pow__(y.array))
        else:
            return PArray(x.array.__pow__(y))

    def __lshift__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__lshift__(y.array))
        else:
            return PArray(x.array.__lshift__(y))

    def __rshift__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__rshift__(y.array))
        else:
            return PArray(x.array.__rshift__(y))

    def __and__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__and__(y.array))
        else:
            return PArray(x.array.__and__(y))

    def __or__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__or__(y.array))
        else:
            return PArray(x.array.__or__(y))

    def __xor__(x, y):
        if isinstance(y, PArray):
            if not x._on_same_device(y):
                raise ValueError("Arrays are not on the same device")
            return PArray(x.array.__xor__(y.array))
        else:
            return PArray(x.array.__xor__(y))

    # Arithmetic, in-place:
    def __iadd__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__iadd__(other.array)
        else:
            self.array.__iadd__(other)
        return self

    def __isub__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__isub__(other.array)
        else:
            self.array.__isub__(other)
        return self

    def __imul__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__imul__(other.array)
        else:
            self.array.__imul__(other)
        return self

    def __idiv__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__idiv__(other.array)
        else:
            self.array.__idiv__(other)
        return self

    def __itruediv__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__itruediv__(other.array)
        else:
            self.array.__itruediv__(other)
        return self

    def __ifloordiv__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ifloordiv__(other.array)
        else:
            self.array.__ifloordiv__(other)
        return self

    def __imod__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__imod__(other.array)
        else:
            self.array.__imod__(other)
        return self

    def __ipow__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ipow__(other.array)
        else:
            self.array.__ipow__(other)
        return self

    def __ilshift__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ilshift__(other.array)
        else:
            self.array.__ilshift__(other)
        return self

    def __irshift__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__irshift__(other.array)
        else:
            self.array.__irshift__(other)
        return self

    def __iand__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__iand__(other.array)
        else:
            self.array.__iand__(other)
        return self

    def __ior__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ior__(other.array)
        else:
            self.array.__ior__(other)
        return self

    def __ixor__(self, other):
        if isinstance(other, PArray):
            if not self._on_same_device(other):
                raise ValueError("Arrays are not on the same device")
            self.array.__ixor__(other.array)
        else:
            self.array.__ixor__(other)
        return self

    # Container customization:

    def __iter__(self):
        return self.array.__iter__()

    def __len__(self):
        return self.array.__len__()

    # Conversion:

    def __int__(self):
        return int(self.array)

    def __float__(self):
        return float(self.array)

    def __complex__(self):
        return complex(self.array)

    def __oct__(self):
        return oct(self.array)

    def __hex__(self):
        return hex(self.array)

    def __bytes__(self):
        return bytes(self.array)

    # String representations:

    def __repr__(self):
        return repr(self._array)

    def __str__(self):
        return str(self._array)

    def __format__(self, format_spec):
        return self._array.__format__(format_spec)

    # Member methods to test C++ PArray instances.

    def get_parray_parentid_from_cpp(self):
        return self._cy_parray.get_parray_parentid()

    def get_num_active_tasks(self, global_dev_id):
        return self._cy_parray.get_num_active_tasks(global_dev_id)
