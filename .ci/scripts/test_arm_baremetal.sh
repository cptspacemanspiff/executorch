#!/bin/bash
# Copyright 2024 Arm Limited and/or its affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

TEST_SUITE=$1

if [[ -z "${TEST_SUITE:-}" ]]; then
  echo "Missing test suite name, exiting..."
  exit 1
else
  echo "Run Arm baremetal test suite ${TEST_SUITE}"
fi

TEST_SUITE_NAME="$(basename "$0") ${TEST_SUITE}"

test_pytest() {
    echo "${TEST_SUITE_NAME}: Run pytest"
    # Run arm baremetal pytest tests without FVP
    pytest -c /dev/null -v -n auto backends/arm/test
}

test_pytest_ethosu_fvp() {
    echo "${TEST_SUITE_NAME}: Run pytest with fvp"
    # Run arm baremetal pytest tests with FVP
    pytest -c /dev/null -v -n auto backends/arm/test -p executorch.backends.arm.test.common --arm_quantize_io --arm_run_corstone300
}

test_run_ethosu_fvp() {
    echo "${TEST_SUITE_NAME}: Test ethos-u delegate examples with run.sh"

    # TOSA quantized
    echo "${TEST_SUITE_NAME}: Test ethos-u target TOSA"
    PYTHON_EXECUTABLE=python bash examples/arm/run.sh --target=TOSA --model_name=mv2
    PYTHON_EXECUTABLE=python bash examples/arm/run.sh --target=TOSA --model_name=esdr
    PYTHON_EXECUTABLE=python bash examples/arm/run.sh --target=TOSA --model_name=emformer_join
    PYTHON_EXECUTABLE=python bash examples/arm/run.sh --target=TOSA --model_name=w2l

    # Ethos-U55
    echo "${TEST_SUITE_NAME}: Test ethos-u target Ethos-U55"
    PYTHON_EXECUTABLE=python bash examples/arm/run.sh --target=ethos-u55-128 --model_name=mv2

    # Ethos-U85
    echo "${TEST_SUITE_NAME}: Test ethos-u target Ethos-U85"
    PYTHON_EXECUTABLE=python bash examples/arm/run.sh --target=ethos-u85-128 --model_name=mv2
    }

${TEST_SUITE}