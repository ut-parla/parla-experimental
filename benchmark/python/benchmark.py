import time
import google_benchmark as benchmark
from google_benchmark import Counter

from modtest import CythonMathWrapper, CppMathWrapper, PythonMath


@benchmark.register(name="Python")
def test(state):
    while state:
        math = PythonMath()
        output = math.add(2, 5)


@benchmark.register(name="CythonMathWrapper")
def test(state):
    while state:
        math = CythonMathWrapper()
        output = math.add(2, 5)


@benchmark.register(name="CppMathWrapper")
def test(state):
    while state:
        math = CppMathWrapper()
        output = math.add(2, 5)


if __name__ == "__main__":
    benchmark.main()
