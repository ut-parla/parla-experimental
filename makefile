all:
	pip install . --verbose

clean:
	python setup.py clean --all

local:
	python setup.py build_ext --inplace

test: local
	. testing/run_tests.sh



