#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

#include "acl/acl.h"
#include "aclnn_complex_transpose.h"

namespace {
constexpr int64_t ROWS = 136;
constexpr int64_t INNER = 8;
constexpr int64_t COLS = 256;

int64_t ParsePositiveArg(int argc, char **argv, int index, int64_t defaultValue)
{
    if (argc <= index) {
        return defaultValue;
    }
    char *end = nullptr;
    const long value = std::strtol(argv[index], &end, 10);
    if (end == argv[index] || value <= 0) {
        return defaultValue;
    }
    return value;
}

int ParseDeviceArg(int argc, char **argv, int index, int defaultValue)
{
    if (argc <= index) {
        return defaultValue;
    }
    char *end = nullptr;
    const long value = std::strtol(argv[index], &end, 10);
    if (end == argv[index] || value < 0) {
        return defaultValue;
    }
    return static_cast<int>(value);
}

bool CheckAcl(aclError ret, const char *message)
{
    if (ret == ACL_SUCCESS) {
        return true;
    }
    std::cerr << message << " failed, ret=" << static_cast<int>(ret) << std::endl;
    return false;
}

void FillInput(std::vector<float> &real, std::vector<float> &imag)
{
    for (size_t i = 0; i < real.size(); ++i) {
        const int64_t centered = static_cast<int64_t>(i % 4096) - 2048;
        real[i] = static_cast<float>(centered) * 0.125f;
        imag[i] = -real[i] + static_cast<float>(i % 7) * 0.25f;
    }
}

void Reference(const std::vector<float> &input, std::vector<float> &output, int64_t batch)
{
    const int64_t lastDim = batch * INNER;
    for (int64_t batchIdx = 0; batchIdx < batch; ++batchIdx) {
        for (int64_t row = 0; row < ROWS; ++row) {
            for (int64_t inner = 0; inner < INNER; ++inner) {
                for (int64_t col = 0; col < COLS; ++col) {
                    const int64_t inputOffset = ((batchIdx * ROWS + row) * INNER + inner) * COLS + col;
                    const int64_t outputOffset = (row * COLS + col) * lastDim + batchIdx * INNER + inner;
                    output[outputOffset] = input[inputOffset];
                }
            }
        }
    }
}

float MaxAbsDiff(const std::vector<float> &lhs, const std::vector<float> &rhs)
{
    float maxDiff = 0.0f;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float diff = std::fabs(lhs[i] - rhs[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
    }
    return maxDiff;
}

aclTensor *CreateTensor(const std::vector<int64_t> &shape, void *devPtr)
{
    return aclCreateTensor(shape.data(), shape.size(), ACL_FLOAT, nullptr, 0,
                           ACL_FORMAT_ND, shape.data(), shape.size(), devPtr);
}

bool CopyToDevice(void *dst, size_t size, const std::vector<float> &src)
{
    return CheckAcl(aclrtMemcpy(dst, size, src.data(), size, ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
}

bool CopyToHost(std::vector<float> &dst, const void *src, size_t size)
{
    return CheckAcl(aclrtMemcpy(dst.data(), size, src, size, ACL_MEMCPY_DEVICE_TO_HOST), "copy output");
}
} // namespace

int main(int argc, char **argv)
{
    const int64_t batch = ParsePositiveArg(argc, argv, 1, 1);
    const int deviceId = ParseDeviceArg(argc, argv, 2, 0);
    const int64_t warmup = ParsePositiveArg(argc, argv, 3, 10);
    const int64_t repeats = ParsePositiveArg(argc, argv, 4, 100);
    const int64_t inputElems = batch * ROWS * INNER * COLS;
    const int64_t outputElems = ROWS * COLS * batch * INNER;
    const size_t inputBytes = static_cast<size_t>(inputElems) * sizeof(float);
    const size_t outputBytes = static_cast<size_t>(outputElems) * sizeof(float);

    std::vector<float> inputR(inputElems);
    std::vector<float> inputI(inputElems);
    std::vector<float> outputR(outputElems, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> outputI(outputElems, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> expectR(outputElems);
    std::vector<float> expectI(outputElems);
    FillInput(inputR, inputI);
    Reference(inputR, expectR, batch);
    Reference(inputI, expectI, batch);

    void *devAR = nullptr;
    void *devAI = nullptr;
    void *devCR = nullptr;
    void *devCI = nullptr;
    void *workspace = nullptr;
    aclrtStream stream = nullptr;
    aclTensor *aRTensor = nullptr;
    aclTensor *aITensor = nullptr;
    aclTensor *cRTensor = nullptr;
    aclTensor *cITensor = nullptr;

    if (!CheckAcl(aclInit(nullptr), "aclInit") ||
        !CheckAcl(aclrtSetDevice(deviceId), "aclrtSetDevice") ||
        !CheckAcl(aclrtCreateStream(&stream), "aclrtCreateStream") ||
        !CheckAcl(aclrtMalloc(&devAR, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc a_r") ||
        !CheckAcl(aclrtMalloc(&devAI, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc a_i") ||
        !CheckAcl(aclrtMalloc(&devCR, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc c_r") ||
        !CheckAcl(aclrtMalloc(&devCI, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST), "malloc c_i") ||
        !CopyToDevice(devAR, inputBytes, inputR) ||
        !CopyToDevice(devAI, inputBytes, inputI)) {
        return 1;
    }

    const std::vector<int64_t> inputShape{batch * ROWS, INNER, COLS};
    const std::vector<int64_t> outputShape{ROWS, COLS, batch * INNER};
    aRTensor = CreateTensor(inputShape, devAR);
    aITensor = CreateTensor(inputShape, devAI);
    cRTensor = CreateTensor(outputShape, devCR);
    cITensor = CreateTensor(outputShape, devCI);
    if (aRTensor == nullptr || aITensor == nullptr || cRTensor == nullptr || cITensor == nullptr) {
        std::cerr << "create tensor failed" << std::endl;
        return 1;
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    aclnnStatus ret = aclnnComplexTransposeGetWorkspaceSize(
        aRTensor, aITensor, cRTensor, cITensor, &workspaceSize, &executor);
    if (ret != ACL_SUCCESS) {
        std::cerr << "aclnnComplexTransposeGetWorkspaceSize failed, ret="
                  << static_cast<int>(ret) << std::endl;
        return 1;
    }
    if (workspaceSize != 0 &&
        !CheckAcl(aclrtMalloc(&workspace, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST), "malloc workspace")) {
        return 1;
    }

    auto launchOp = [&]() -> bool {
        const aclnnStatus launchRet = aclnnComplexTranspose(workspace, workspaceSize, executor, stream);
        if (launchRet != ACL_SUCCESS) {
            std::cerr << "aclnnComplexTranspose failed, ret=" << static_cast<int>(launchRet) << std::endl;
            return false;
        }
        return true;
    };

    for (int64_t i = 0; i < warmup; ++i) {
        if (!launchOp()) {
            return 1;
        }
    }
    if (!CheckAcl(aclrtSynchronizeStream(stream), "warmup synchronize")) {
        return 1;
    }

    aclrtEvent startEvent = nullptr;
    aclrtEvent endEvent = nullptr;
    if (!CheckAcl(aclrtCreateEvent(&startEvent), "create start event") ||
        !CheckAcl(aclrtCreateEvent(&endEvent), "create end event")) {
        (void)aclrtDestroyEvent(startEvent);
        (void)aclrtDestroyEvent(endEvent);
        return 1;
    }

    std::vector<float> elapsedMs;
    elapsedMs.reserve(static_cast<size_t>(repeats));
    for (int64_t i = 0; i < repeats; ++i) {
        if (!CheckAcl(aclrtRecordEvent(startEvent, stream), "record start event") ||
            !launchOp() ||
            !CheckAcl(aclrtRecordEvent(endEvent, stream), "record end event") ||
            !CheckAcl(aclrtSynchronizeStream(stream), "timed synchronize")) {
            (void)aclrtDestroyEvent(startEvent);
            (void)aclrtDestroyEvent(endEvent);
            return 1;
        }
        float oneElapsedMs = 0.0f;
        if (!CheckAcl(aclrtEventElapsedTime(&oneElapsedMs, startEvent, endEvent), "event elapsed time")) {
            (void)aclrtDestroyEvent(startEvent);
            (void)aclrtDestroyEvent(endEvent);
            return 1;
        }
        elapsedMs.push_back(oneElapsedMs);
    }
    (void)aclrtDestroyEvent(startEvent);
    (void)aclrtDestroyEvent(endEvent);

    if (!CheckAcl(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream") ||
        !CopyToHost(outputR, devCR, outputBytes) ||
        !CopyToHost(outputI, devCI, outputBytes)) {
        return 1;
    }

    const float diffR = MaxAbsDiff(outputR, expectR);
    const float diffI = MaxAbsDiff(outputI, expectI);
    double totalMs = 0.0;
    for (float value : elapsedMs) {
        totalMs += value;
    }
    const float minMs = *std::min_element(elapsedMs.begin(), elapsedMs.end());
    const float maxMs = *std::max_element(elapsedMs.begin(), elapsedMs.end());
    const double avgMs = totalMs / static_cast<double>(elapsedMs.size());
    std::cout << "batch=" << batch << " device=" << deviceId
              << " workspace=" << workspaceSize
              << " warmup=" << warmup
              << " repeats=" << repeats
              << " avg_ms=" << avgMs
              << " min_ms=" << minMs
              << " max_ms=" << maxMs
              << " max_diff_r=" << diffR
              << " max_diff_i=" << diffI << std::endl;

    if (workspace != nullptr) {
        (void)aclrtFree(workspace);
    }
    (void)aclDestroyTensor(aRTensor);
    (void)aclDestroyTensor(aITensor);
    (void)aclDestroyTensor(cRTensor);
    (void)aclDestroyTensor(cITensor);
    (void)aclrtFree(devAR);
    (void)aclrtFree(devAI);
    (void)aclrtFree(devCR);
    (void)aclrtFree(devCI);
    (void)aclrtDestroyStream(stream);
    (void)aclrtResetDevice(deviceId);
    (void)aclFinalize();

    return (diffR == 0.0f && diffI == 0.0f) ? 0 : 1;
}
