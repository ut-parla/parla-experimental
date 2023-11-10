all:
	pip install . --verbose

clean:
	rm -rf build

test: local
	. testing/run_tests.sh



