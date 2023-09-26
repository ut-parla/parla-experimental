"""!
@file threads.py
@brief Helper classes for managing threads in testing applications
"""
import threading


class Propagate(threading.Thread):
    """!
    @brief threading.Thread wrapper that propagates caught exceptions back to the main thread. 

    Useful for testing race conditions and timeouts in pytest.
    """

    def __init__(self, target, args=None):
        super().__init__(target=target, args=args)
        self.ex = None
        self.value = None

    def run(self):
        try:
            if self._args is None:
                self.value = self._target()
            else:
                self.value = self._target(*self._args)
        except BaseException as e:
            self.ex = e

    def join(self, timeout=None):
        super().join(timeout)
        if self.ex is not None:
            raise self.ex
