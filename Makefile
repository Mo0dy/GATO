PLANTS := indy7 iiwa14
KNOTS := 8 16 32 64 128
PYTHON_EXTENSION_TARGETS := $(foreach plant,$(PLANTS),$(foreach knot,$(KNOTS),build-$(plant)-n$(knot)))

.PHONY: help build build-fresh build-bsqp $(PYTHON_EXTENSION_TARGETS) test-cuda clean-build

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

clean-build:
	rm -rf "$${GATO_BUILD_DIR:-build}"
