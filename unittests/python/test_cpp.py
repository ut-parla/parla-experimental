from modtest import CppMathWrapper as MathWrapper
import pytest


@pytest.mark.parametrize('input, expected', [(5, 7), (6, 8)])
def test_add(input, expected):
    math = MathWrapper()
    output = math.add(2, input)
    assert output == expected


@pytest.mark.parametrize('input, expected', [(5, 3), (6, 4)])
def test_sub(input, expected):
    math = MathWrapper()
    output = math.sub(input, 2)
    assert output == expected
