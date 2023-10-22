all:
	python3 -m pip install . --verbose

clean:
	python2 setup.py clean --all
	rm -rf .mkdoxy
	rm -rf docs/doxygen
	rm -rf site

local:
	python3 setup.py build_ext --inplace

test: local
	. testing/run_tests.sh



