#!/bin/bash
set -eo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
BUILD_DIR="${SCRIPT_DIR}/build"

source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
export DDK_PATH=${DDK_PATH:-/usr/local/Ascend/ascend-toolkit/latest}

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"
cmake ..
cmake --build . -j

"${SCRIPT_DIR}/output/execute_complex_transpose" "${1:-1}" "${2:-0}"
