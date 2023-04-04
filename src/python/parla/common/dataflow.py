from crosspy import CrossPyArray

from parla.common.parray.core import PArray

from typing import List, Any, Tuple, Union
from itertools import chain


class DataflowIterator:
    """
    Itrator class for Dataflow.
    """
    def __init__(self, df):
        self._df = df
        self._idx = 0

    def __next__(self):
        """
        Return the next value from Dataflow's data lists:
        input -> output -> in/output lists
        """
        if self._idx < (len(self._df._input) + len(self._df._output) +
                        len(self._df._inout)):
            if self._idx < len(self._df._input):
                # First, iterate input data operands.
                cur_item = self._df._input[self._idx][0]
            elif self._idx < (len(self._df._input) + len(self._df._output)):
                # Second, iterate output data operands.
                cur_item = self._df._output[self._idx - len(self._df._input)][0]
            else:
                # Third, iterate input/output data operands.
                cur_item = self._df._inout[self._idx - len(self._df._input) -
                                           len(self._df._output)][0]
            self._idx += 1
            return cur_item
        raise StopIteration


class Dataflow:
    """
    This class manages a dataflow of a task.
    The dataflow consists of input, output, and input/output PArrays.
    The dataflow can be iterated through DataflowIterator.
    """

    def __init__(self, input: List[Union[CrossPyArray, Tuple[PArray, int]]],
                 output: List[Union[CrossPyArray, Tuple[PArray, int]]],
                 inout: List[Union[CrossPyArray, Tuple[PArray, int]]]):
        self._input = self.process_crosspys(input)
        self._output = self.process_crosspys(output)
        self._inout = self.process_crosspys(inout)

    @property
    def input(self) -> List:
        if self._input == None:
            return []
        return self._input

    @property
    def output(self) -> List:
        if self._output == None:
            return []
        return self._output

    @property
    def inout(self) -> List:
        if self._inout == None:
            return []
        return self._inout

    def process_crosspys(
        self, _in: List[Union[CrossPyArray, Tuple[PArray, int]]]) \
            -> List[Tuple[PArray, int]]:
        """
        Check elements of IN/OUT/INOUT parameters in @spawn
        and convert to Tuple[PArray, int] if an element is a CrossPyArray
        """
        _out = []
        if _in is not None:
            for element in _in:
                if isinstance(element, tuple):
                    assert isinstance(element[0], PArray)
                    assert isinstance(element[1], int)
                    _out.append(element)
                elif isinstance(element, CrossPyArray):
                    # A Crosspy's partition number is corresponding
                    # to an order of the placement.
                    for i, parray_list in enumerate(element.device_view()):
                        for parray in parray_list:
                            if isinstance(parray, PArray):
                                # Skip CuPy or NumPy arrays.
                                _out.element((i, parray))
                else:
                    raise TypeError("IN/OUT/INOUT should be either a tuple of PArray",
                                    "and its partition number or CrossPyArray")
        return _out

    def __iter__(self):
        return DataflowIterator(self)
