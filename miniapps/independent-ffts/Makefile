all:
	pip install . --verbose

clean:
	rm -rf _skbuild
	rm -rf .pytest_cache
	rm -rf src/cufftmg.egg-info
	python setup.py clean --all

local:
	python setup.py build_ext --inplace

test: local
	. test/run_tests.sh
