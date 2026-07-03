#!/bin/bash
CURRENT_DIR=$(
    cd "$(dirname "${BASH_SOURCE:-$0}")"
    pwd
)

BATCH=${1:-1}
DEVICE_ARG=${2:-}
WARMUP=${3:-10}
REPEATS=${4:-100}

if [ -n "$ASCEND_INSTALL_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
elif [ -n "$ASCEND_HOME_PATH" ]; then
    _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
else
    if [ -d "$HOME/Ascend/ascend-toolkit/latest" ]; then
        _ASCEND_INSTALL_PATH=$HOME/Ascend/ascend-toolkit/latest
    else
        _ASCEND_INSTALL_PATH=/usr/local/Ascend/ascend-toolkit/latest
    fi
fi

if [ -f "$_ASCEND_INSTALL_PATH/bin/setenv.bash" ]; then
    source "$_ASCEND_INSTALL_PATH/bin/setenv.bash"
elif [ -f "$_ASCEND_INSTALL_PATH/set_env.sh" ]; then
    source "$_ASCEND_INSTALL_PATH/set_env.sh"
else
    echo "ERROR: Ascend environment script not found under $_ASCEND_INSTALL_PATH"
    exit 1
fi

export DDK_PATH=$_ASCEND_INSTALL_PATH
export NPU_HOST_LIB=$_ASCEND_INSTALL_PATH/$(arch)-$(uname -s | tr '[:upper:]' '[:lower:]')/lib64
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
if [ -n "$DEVICE_ARG" ]; then
    export DEVICE_ID=$DEVICE_ARG
fi
if [ -z "$DEVICE_ID" ] && [ -z "$ASCEND_DEVICE_ID" ]; then
    export DEVICE_ID=0
fi
export ASCEND_DEVICE_ID=${ASCEND_DEVICE_ID:-$DEVICE_ID}
echo "INFO: use DEVICE_ID=${DEVICE_ID:-$ASCEND_DEVICE_ID}"

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1
import numpy
PY
then
    if [ -x /usr/local/python3.12.13/bin/python3 ]; then
        PYTHON_BIN=/usr/local/python3.12.13/bin/python3
    fi
fi
echo "INFO: use PYTHON_BIN=$PYTHON_BIN"

function main {
    if [ -d "$HOME/ascend/log" ]; then
        rm -rf "$HOME"/ascend/log/*
    fi
    mkdir -p "$CURRENT_DIR"/input "$CURRENT_DIR"/output
    rm -f "$CURRENT_DIR"/input/*.bin
    rm -f "$CURRENT_DIR"/output/*.bin
    rm -f "$CURRENT_DIR"/output/execute_complex_transpose
    find "$CURRENT_DIR/src" "$CURRENT_DIR/inc" -type f -exec touch {} +

    cd "$CURRENT_DIR"
    "$PYTHON_BIN" scripts/gen_data.py --batch "$BATCH"
    if [ $? -ne 0 ]; then
        echo "ERROR: generate input data failed!"
        return 1
    fi
    echo "INFO: generate input data success!"

    cd "$CURRENT_DIR"
    rm -rf build
    mkdir -p build
    cd build
    cmake ../src -DCMAKE_SKIP_RPATH=TRUE
    if [ $? -ne 0 ]; then
        echo "ERROR: cmake failed!"
        return 1
    fi
    echo "INFO: cmake success!"
    make -j
    if [ $? -ne 0 ]; then
        echo "ERROR: make failed!"
        return 1
    fi
    echo "INFO: make success!"

    export LD_LIBRARY_PATH=$_ASCEND_INSTALL_PATH/opp/vendors/customize/op_api/lib:/usr/local/Ascend/driver/lib64/driver:$LD_LIBRARY_PATH
    cd "$CURRENT_DIR/output"
    echo "INFO: execute op!"
    ./execute_complex_transpose "$BATCH" "${DEVICE_ID:-$ASCEND_DEVICE_ID}" "$WARMUP" "$REPEATS"
    if [ $? -ne 0 ]; then
        echo "ERROR: acl executable run failed! please check your project!"
        return 1
    fi
    echo "INFO: acl executable run success!"

    cd "$CURRENT_DIR"
    "$PYTHON_BIN" scripts/verify_result.py --batch "$BATCH"
    if [ $? -ne 0 ]; then
        echo "ERROR: verify result failed!"
        return 1
    fi
}

main
