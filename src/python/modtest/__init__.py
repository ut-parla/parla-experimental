from .cython import core
from .common import PythonMath

CythonMathWrapper = core.CythonMathWrapper
CppMathWrapper = core.CppMathWrapper
__all__ = ['PythonMath', 'CythonMathWrapper', 'CppMathWrapper']
