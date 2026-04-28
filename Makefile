PLANTS := indy7 iiwa14
KNOTS := 8 16 32 64 128
PYTHON_EXTENSION_TARGETS := $(foreach plant,$(PLANTS),$(foreach knot,$(KNOTS),build-$(plant)-n$(knot)))

.PHONY: help build build-fresh build-bsqp $(PYTHON_EXTENSION_TARGETS) test test-fast test-python test-tracking test-performance test-cuda clean-build

help:
	@printf '%s\n' 'GATO build targets:'
	@printf '  %-22s %s\n' 'build' 'Incremental full build'
	@printf '  %-22s %s\n' 'build-fresh' 'Clean full build'
	@printf '  %-22s %s\n' 'build-bsqp' 'Build C++ example target'
	@for plant in $(PLANTS); do \
		for knot in $(KNOTS); do \
			printf '  %-22s %s\n' "build-$${plant}-n$${knot}" 'Build Python extension target'; \
		done; \
	done
	@printf '  %-22s %s\n' 'test-cuda' 'Validate built CUDA artifacts'
	@printf '  %-22s %s\n' 'test' 'Build test dependencies and run CTest'
	@printf '  %-22s %s\n' 'test-fast' 'Run fast Python behavior tests'
	@printf '  %-22s %s\n' 'test-python' 'Run fast Python behavior tests'
	@printf '  %-22s %s\n' 'test-tracking' 'Build and run short tracking smoke test'
	@printf '  %-22s %s\n' 'test-performance' 'Build and write tracking performance artifacts'
	@printf '  %-22s %s\n' 'clean-build' 'Remove build directory'

build:
	./tools/build.sh

build-fresh:
	./tools/build.sh --fresh

build-bsqp:
	./tools/build.sh --target bsqp

define PYTHON_EXTENSION_RULE
build-$(1)-n$(2):
	./tools/build.sh --target bsqpN$(2)_$(1)
endef

$(foreach plant,$(PLANTS),$(foreach knot,$(KNOTS),$(eval $(call PYTHON_EXTENSION_RULE,$(plant),$(knot)))))

test-cuda:
	./tools/test_cuda_compatibility.sh --strict --build-dir "$${GATO_BUILD_DIR:-build}"

test:
	./tools/build.sh --target check

test-fast:
	./tools/build.sh --target check-fast

test-python:
	./tools/build.sh --target check-python

test-tracking:
	./tools/build.sh --target check-tracking

test-performance:
	./tools/build.sh --target check-performance

clean-build:
	rm -rf "$${GATO_BUILD_DIR:-build}"
